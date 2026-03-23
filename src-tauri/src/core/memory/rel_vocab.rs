pub const CANONICAL_RELATIONS: &[&str] = &[
    "created_by",
    "owns",
    "works_at",
    "works_with",
    "collaborates_with",
    "prefers",
    "likes",
    "dislikes",
    "project_member_of",
    "member_of",
    "lives_in",
    "friends",
    "parent_of",
    "father_of",
    "mother_of",
    "spouse_of",
    "sibling_of",
    "employer_of",
    "writes",
    "depends_on",
    "causes",
    "improves",
    "blocks",
    "related_to",
    "implemented_in",
];

pub fn is_canonical_relation(rel_type: &str) -> bool {
    let rel = rel_type.trim().to_lowercase();
    CANONICAL_RELATIONS.iter().any(|r| *r == rel)
}
