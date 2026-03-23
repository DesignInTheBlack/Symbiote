// Memory System Types (ICS v4.1)

export interface ScoredFact {
    id: number;
    entity_label: string;
    key: string;
    topic_key: string;
    value: string;
    confidence: number;
    score: number;
    time_bucket_kind: string;
    time_bucket_value?: string;
    signature_hash: string;
    polarity: string;
    scope: string;
    observed_at?: string;
    observed_at_formatted?: string;
}

export interface ScoredRel {
    id: number;
    rel_type: string;
    participants: ScoredRelParticipant[];
    direction?: string | null;
    order_is_trusted?: boolean;
    confidence: number;
    score: number;
    time_bucket_kind: string;
    time_bucket_value?: string;
    signature_hash: string;
    polarity: string;
    scope: string;
    observed_at?: string;
    observed_at_formatted?: string;
}

export interface ScoredRelParticipant {
    role: string;
    entity_id: number;
    entity_label: string;
}

export interface ConflictSet {
    id: number;
    topic_key: string;
    status: 'open' | 'resolved' | 'archived';
    priority: string;
    resolution_note: string | null;
    created_at: string;
    updated_at: string;
}

export interface ConflictMemberView {
    belief_id: number;
    kind: string;
    polarity: string;
    confidence: number;
    status: string;
    preview: string;
    observed_at?: string | null;
    evidence_snippet?: string | null;
}

export interface ConflictView {
    id: number;
    topic_key: string;
    status: string;
    priority: string;
    resolution_note?: string | null;
    created_at: string;
    updated_at: string;
    members: ConflictMemberView[];
}

export interface MemoryPacket {
    facts: ScoredFact[];
    relations: ScoredRel[];
    conflicts: ConflictSet[];
    bound_handles: Record<string, string>;
    shadowed_by_scope_count?: number;
    dropped_by_scope_count?: number;
}

export interface ClaimOutcome {
    claim_id: string;
    status: string;
    reason?: string | null;
    scope: string;
    session_id?: string | null;
    created_at: string;
}

export interface CompileResult {
    written_ids: number[];
    conflict_ids: number[];
    pending_writes: PendingWrite[];
    claim_ids?: string[];
    errors: string[];
}

export interface PendingWrite {
    id: number;
    parsed_lines: string;
    candidates_json: string;
    status: string;
    created_at: string;
}

export interface ConsolidationResult {
    aliases_promoted: number;
    sketches_updated: number;
    stale_deactivated: number;
    conflicts_archived: number;
}

// 3D Graph Types
export interface GraphNode {
    id: string;
    nodeType: 'entity' | 'fact' | 'relation' | 'conflict';
    label: string;
    entityType?: string;
    confidence?: number;
    accessCount: number;
    activation?: number;  // Working set activation (0.0-1.0)
    lastAccessed?: string;
    key?: string; // For facts
    value?: string; // For facts
    // Phase 4: Temporal and scope fields
    scope?: string;
    polarity?: string;
    timeBucketKind?: string;
    timeBucketValue?: string;
    // Relation participants (role, entity_name)
    participants?: [string, string][];
    x?: number;
    y?: number;
    z?: number;
}

export interface GraphLink {
    source: string;
    target: string;
    linkType: string;
    label?: string;
    strength: number;
}

export interface MemoryGraph {
    nodes: GraphNode[];
    links: GraphLink[];
}

export interface EpisodicEvent {
    id: string;
    event_type: string;
    payload: any;
    timestamp: string;
    run_id?: string | null;
    trace_id?: string | null;
    conversation_id?: string | null;
    scope?: string | null;
    source_type: string;
    source_ref?: string | null;
    linked_belief_id?: number | null;
    linked_artifact_id?: string | null;
}

export interface AnchorDebug {
    entity_id: number;
    label: string;
    source: string;
    score: number;
}

export interface TraversalDebug {
    starting_anchors: number;
    hops_taken: number;
    entities_visited: number;
    beliefs_collected: number;
    frontier_max_size: number;
    was_bounded: boolean;
}

export interface RankingComponents {
    evidence_weight: number;
    confidence: number;
    salience: number;
    time_decay: number;
    assoc_weight: number;
    i4_support: number;
}

export interface FactRankingDebug {
    id: number;
    topic_key: string;
    value_preview: string;
    score: number;
    components: RankingComponents;
}

export interface SelectionDebug {
    facts_before_filter: number;
    facts_after_negation: number;
    facts_after_shadowing: number;
    facts_final: number;
    top_facts: FactRankingDebug[];
}

export interface ShadowedBelief {
    id: number;
    topic_key: string;
    scope: string;
    shadowed_by_scope: string;
}

export interface ShadowingDebug {
    topic_groups_count: number;
    beliefs_shadowed: number;
    shadowed_details: ShadowedBelief[];
}

export interface RetrievalDebugLog {
    query: string;
    anchors: AnchorDebug[];
    traversal: TraversalDebug;
    selection: SelectionDebug;
    shadowing: ShadowingDebug;
    duration_ms: number;
}

export interface MemoryDebugLog {
    retrieval?: RetrievalDebugLog;
    resolution?: any;
    compile?: any;
    timestamp: string;
}

export interface MemoryRetrievalDebugResponse {
    summary: string;
    log: MemoryDebugLog;
}
