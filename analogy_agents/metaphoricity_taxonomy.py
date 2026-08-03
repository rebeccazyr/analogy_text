from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analogy_agents.schemas import MTaxonomyJudgment


M_TAXONOMY_POLICY_VERSION = "m_v9_fair_lca_v2"
M_TAXONOMY_CACHE_NAMESPACE = "m_v9_fair_lca_mapper_v2"
M_TAXONOMY_AGENT_VERSION = "m_v10_taxonomy_agent_final_v1"
M_TAXONOMY_AGENT_CACHE_NAMESPACE = "m_v10_taxonomy_agent_final_v1"
M_CONCEPTUAL_DISTANCE_VERSION = "m_v11_overall_conceptual_distance_v1"
M_CONCEPTUAL_DISTANCE_CACHE_NAMESPACE = (
    "m_v11_overall_conceptual_distance_v1"
)
M_CONCEPTUAL_DISTANCE_CRITIC_VERSION = (
    "m_v12_conceptual_distance_critic_adjudicator_v1"
)
M_CONCEPTUAL_DISTANCE_CRITIC_CACHE_NAMESPACE = (
    "m_v12_conceptual_distance_critic_adjudicator_v1"
)
M_TAXONOMY_EVIDENCE_VERSION = "m_taxonomy_three_axis_evidence_v1"
DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "metaphoricity_taxonomy.json"
)


@dataclass(frozen=True)
class TaxonomyNode:
    node_id: str
    parent: str | None
    level: int
    selectable: bool
    label: str
    definition: str


class TaxonomyTree:
    def __init__(self, name: str, payload: dict[str, Any]):
        self.name = name
        self.nodes = {
            node_id: TaxonomyNode(
                node_id=node_id,
                parent=node_payload.get("parent"),
                level=int(node_payload["level"]),
                selectable=bool(node_payload["selectable"]),
                label=node_payload["label"],
                definition=node_payload["definition"],
            )
            for node_id, node_payload in payload.items()
        }
        roots = [node.node_id for node in self.nodes.values() if node.parent is None]
        if len(roots) != 1:
            raise ValueError(f"{name} taxonomy must contain exactly one root")
        self.root_id = roots[0]
        if self.nodes[self.root_id].level != 0:
            raise ValueError(f"{name} taxonomy root must be level 0")
        for node in self.nodes.values():
            if node.parent is None:
                continue
            if node.parent not in self.nodes:
                raise ValueError(
                    f"{name} taxonomy node {node.node_id} has unknown parent "
                    f"{node.parent}"
                )
            parent = self.nodes[node.parent]
            if node.level != parent.level + 1:
                raise ValueError(
                    f"{name} taxonomy node {node.node_id} must be exactly one "
                    f"level below {node.parent}"
                )
            self._lineage(node.node_id)
        invalid_selectable = [
            node.node_id
            for node in self.nodes.values()
            if node.selectable and node.level != 3
        ]
        if invalid_selectable:
            raise ValueError(
                f"Selectable {name} nodes must all be level 3: "
                f"{invalid_selectable}"
            )

    def _lineage(self, node_id: str) -> list[str]:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown {self.name} taxonomy node: {node_id}")
        lineage: list[str] = []
        seen: set[str] = set()
        current: str | None = node_id
        while current is not None:
            if current in seen:
                raise ValueError(f"Cycle in {self.name} taxonomy at {current}")
            seen.add(current)
            lineage.append(current)
            current = self.nodes[current].parent
        return lineage

    def validate_selectable(self, node_id: str) -> None:
        self._lineage(node_id)
        if not self.nodes[node_id].selectable:
            raise ValueError(
                f"{self.name} taxonomy node {node_id} is not a selectable leaf"
            )

    def lowest_common_ancestor(self, left: str, right: str) -> TaxonomyNode:
        left_ancestors = set(self._lineage(left))
        for node_id in self._lineage(right):
            if node_id in left_ancestors:
                return self.nodes[node_id]
        raise ValueError(f"No common ancestor for {left} and {right}")

    def common_parent_level(self, left: str, right: str) -> int:
        return self.lowest_common_ancestor(left, right).level

    def compact_nodes(self) -> list[dict[str, Any]]:
        return [
            {
                "id": node.node_id,
                "parent": node.parent,
                "level": node.level,
                "selectable": node.selectable,
                "label": node.label,
                "definition": node.definition,
            }
            for node in self.nodes.values()
        ]


class MTaxonomy:
    def __init__(self, payload: dict[str, Any]):
        self.version = payload["version"]
        self.construction = payload["construction"]
        self.policy = payload["policy"]
        self.domains = TaxonomyTree("domain", payload["domains"])
        self.entities = TaxonomyTree("entity", payload["entities"])
        self.relations = TaxonomyTree("relation", payload["relations"])
        self.applicability = payload["applicability"]
        self.target_profiles = payload["target_profiles"]
        self._validate_profiles()

    @classmethod
    def load(cls, path: Path = DEFAULT_TAXONOMY_PATH) -> "MTaxonomy":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _validate_profiles(self) -> None:
        for target, profile in self.target_profiles.items():
            self.domains.validate_selectable(profile["domain_path"])
            for relation_path in profile["relation_paths"]:
                self.relations.validate_selectable(relation_path)
            if profile["scope_type"] not in {
                "general_formal_or_practice",
                "domain_specific",
            }:
                raise ValueError(f"Invalid scope_type for target {target}")

    def target_profile(self, target: str) -> dict[str, Any]:
        try:
            return self.target_profiles[target]
        except KeyError as error:
            raise ValueError(
                f"Target {target!r} is not covered by taxonomy {self.version}"
            ) from error

    def prompt_payload(self, target: str) -> dict[str, Any]:
        return {
            "version": self.version,
            "construction": self.construction,
            "target_profile": self.target_profile(target),
            "applicability": {
                name: {"definition": payload["definition"]}
                for name, payload in self.applicability.items()
            },
            "domain_nodes": self.domains.compact_nodes(),
            "entity_nodes": self.entities.compact_nodes(),
            "relation_nodes": self.relations.compact_nodes(),
        }

    def validate_judgment(
        self,
        target: str,
        judgment: MTaxonomyJudgment,
    ) -> None:
        profile = self.target_profile(target)
        if judgment.applicability not in self.applicability:
            raise ValueError(
                f"Unknown applicability value: {judgment.applicability}"
            )
        self.domains.validate_selectable(judgment.source_domain_path)
        self.relations.validate_selectable(judgment.source_relation_path)
        self.relations.validate_selectable(judgment.target_relation_path)
        if judgment.target_relation_path not in profile["relation_paths"]:
            raise ValueError(
                f"Target relation {judgment.target_relation_path} is not allowed "
                f"for {target}"
            )
        for mapping in judgment.role_mappings:
            self.entities.validate_selectable(mapping.source_entity_path)
            self.entities.validate_selectable(mapping.target_entity_path)

    def score_trace(
        self,
        target: str,
        judgment: MTaxonomyJudgment,
        literal_instance: str,
    ) -> dict[str, Any]:
        self.validate_judgment(target, judgment)
        profile = self.target_profile(target)
        applicability_payload = self.applicability[judgment.applicability]

        domain_lca = self.domains.lowest_common_ancestor(
            judgment.source_domain_path,
            profile["domain_path"],
        )
        relation_lca = self.relations.lowest_common_ancestor(
            judgment.source_relation_path,
            judgment.target_relation_path,
        )
        role_lcas = [
            {
                "source_role": mapping.source_role,
                "target_role": mapping.target_role,
                "source_entity_path": mapping.source_entity_path,
                "target_entity_path": mapping.target_entity_path,
                "common_parent_id": self.entities.lowest_common_ancestor(
                    mapping.source_entity_path,
                    mapping.target_entity_path,
                ).node_id,
                "common_parent_level": self.entities.common_parent_level(
                    mapping.source_entity_path,
                    mapping.target_entity_path,
                ),
            }
            for mapping in judgment.role_mappings
        ]
        level_0_roles = [
            role for role in role_lcas if role["common_parent_level"] == 0
        ]
        level_1_roles = [
            role for role in role_lcas if role["common_parent_level"] == 1
        ]

        literal_applicability = (
            bool(applicability_payload["literal"])
            and literal_instance == "yes"
        )
        if judgment.applicability == "specialization_of":
            literal_applicability = literal_applicability and (
                profile["scope_type"] == "general_formal_or_practice"
            )
        if profile["scope_type"] == "domain_specific":
            literal_applicability = literal_applicability and (
                domain_lca.level
                >= int(
                    self.policy[
                        "literal_domain_lca_level_for_specific_target"
                    ]
                )
            )

        if literal_applicability:
            score = 0
            decisive_rule = "literal_applicability"
        elif (
            relation_lca.level >= int(self.policy["m1_relation_lca_level"])
            and not level_0_roles
            and len(level_1_roles)
            <= int(self.policy["m1_max_level_1_roles"])
        ):
            score = 1
            decisive_rule = "shared_relation_family_and_role_parent_levels"
        else:
            score = 2
            decisive_rule = "cross_kind_projection"

        return {
            "version": M_TAXONOMY_POLICY_VERSION,
            "taxonomy_version": self.version,
            "score": score,
            "decisive_rule": decisive_rule,
            "applicability": judgment.applicability,
            "literal_instance_judge": literal_instance,
            "literal_applicability_accepted": literal_applicability,
            "source_domain_path": judgment.source_domain_path,
            "target_domain_path": profile["domain_path"],
            "domain_common_parent_id": domain_lca.node_id,
            "domain_common_parent_level": domain_lca.level,
            "source_relation_path": judgment.source_relation_path,
            "target_relation_path": judgment.target_relation_path,
            "relation_common_parent_id": relation_lca.node_id,
            "relation_common_parent_level": relation_lca.level,
            "role_common_parents": role_lcas,
            "level_0_role_count": len(level_0_roles),
            "level_1_role_count": len(level_1_roles),
            "policy": self.policy,
            "construction": self.construction,
        }

    def comparison_trace(
        self,
        target: str,
        judgment: MTaxonomyJudgment,
    ) -> dict[str, Any]:
        """Return three-axis taxonomy evidence without assigning an M score."""
        self.validate_judgment(target, judgment)
        profile = self.target_profile(target)
        source_domain = self.domains.nodes[judgment.source_domain_path]
        target_domain = self.domains.nodes[profile["domain_path"]]
        domain_lca = self.domains.lowest_common_ancestor(
            judgment.source_domain_path,
            profile["domain_path"],
        )
        source_relation = self.relations.nodes[
            judgment.source_relation_path
        ]
        target_relation = self.relations.nodes[
            judgment.target_relation_path
        ]
        relation_lca = self.relations.lowest_common_ancestor(
            judgment.source_relation_path,
            judgment.target_relation_path,
        )
        role_comparisons = []
        for mapping in judgment.role_mappings:
            source_entity = self.entities.nodes[mapping.source_entity_path]
            target_entity = self.entities.nodes[mapping.target_entity_path]
            common_parent = self.entities.lowest_common_ancestor(
                mapping.source_entity_path,
                mapping.target_entity_path,
            )
            role_comparisons.append(
                {
                    "source_role": mapping.source_role,
                    "target_role": mapping.target_role,
                    "source_entity": {
                        "path": source_entity.node_id,
                        "label": source_entity.label,
                        "definition": source_entity.definition,
                    },
                    "target_entity": {
                        "path": target_entity.node_id,
                        "label": target_entity.label,
                        "definition": target_entity.definition,
                    },
                    "common_parent": {
                        "path": common_parent.node_id,
                        "label": common_parent.label,
                        "level": common_parent.level,
                    },
                    "mapping_evidence": mapping.evidence,
                }
            )

        return {
            "version": M_TAXONOMY_EVIDENCE_VERSION,
            "taxonomy_version": self.version,
            "level_semantics": {
                "3": "same selectable fine-grained subtype",
                "2": "different leaves in the same specific family",
                "1": "same broad category but different specific families",
                "0": "only the taxonomy root is shared",
            },
            "domain_axis": {
                "target_scope_type": profile["scope_type"],
                "mapper_applicability": judgment.applicability,
                "applicability_evidence": judgment.applicability_evidence,
                "source": {
                    "path": source_domain.node_id,
                    "label": source_domain.label,
                    "definition": source_domain.definition,
                },
                "target": {
                    "path": target_domain.node_id,
                    "label": target_domain.label,
                    "definition": target_domain.definition,
                },
                "common_parent": {
                    "path": domain_lca.node_id,
                    "label": domain_lca.label,
                    "level": domain_lca.level,
                },
            },
            "relation_axis": {
                "source": {
                    "path": source_relation.node_id,
                    "label": source_relation.label,
                    "definition": source_relation.definition,
                },
                "target": {
                    "path": target_relation.node_id,
                    "label": target_relation.label,
                    "definition": target_relation.definition,
                },
                "common_parent": {
                    "path": relation_lca.node_id,
                    "label": relation_lca.label,
                    "level": relation_lca.level,
                },
            },
            "entity_role_axis": role_comparisons,
        }


def load_m_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> MTaxonomy:
    return MTaxonomy.load(path)
