import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import SpriteText from 'three-spritetext';
import * as THREE from 'three';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { invokeWithTimeout } from '../utils/tauri';
import { getCssVar, THEME_CHANGE_EVENT } from '../utils/theme';
import { MemoryGraph, GraphNode, GraphLink, CompileResult, EpisodicEvent, MemoryRetrievalDebugResponse, ClaimOutcome } from '../types/memoryTypes';
import { createHologramMaterial } from './memory_graph/hologramShader';
import { createTechTexture } from './memory_graph/textureGenerator';

interface Props {
    isOpen: boolean;
    onClose: () => void;
}

interface ContextMenuState {
    x: number;
    y: number;
    node: GraphNode;
}

const EPISODIC_LIMIT = 80;
const PROVENANCE_LIMIT = 8;
const MEMORY_GRAPH_TIMEOUT_MS = 15000;
const MEMORY_WRITE_TIMEOUT_MS = 20000;

const rgbaFrom = (color: string, alpha: number) => {
    const parsed = new THREE.Color(color);
    const r = Math.round(parsed.r * 255);
    const g = Math.round(parsed.g * 255);
    const b = Math.round(parsed.b * 255);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const readGraphTheme = () => ({
    node: {
        conflict: getCssVar('--graph-node-conflict', '#ff3d00'),
        person: getCssVar('--graph-node-person', '#00e676'),
        place: getCssVar('--graph-node-place', '#00b0ff'),
        work: getCssVar('--graph-node-work', '#ff9100'),
        concept: getCssVar('--graph-node-concept', '#d500f9'),
        event: getCssVar('--graph-node-event', '#ff4081'),
        project: getCssVar('--graph-node-project', '#7c4dff'),
        system: getCssVar('--graph-node-system', '#00bcd4'),
        fallback: getCssVar('--graph-node-default', '#78909c'),
        muted: getCssVar('--graph-node-muted', '#607d8b'),
    },
    link: {
        fact: getCssVar('--graph-link-fact', '#b0bec5'),
        supersedes: getCssVar('--graph-link-supersedes', '#ffab40'),
        contradicts: getCssVar('--graph-link-contradicts', '#ff5252'),
        supports: getCssVar('--graph-link-supports', '#69f0ae'),
        derived: getCssVar('--graph-link-derived', '#ea80fc'),
        relation: getCssVar('--graph-link-relation', '#81d4fa'),
        other: getCssVar('--graph-link-other', '#ff4081'),
    },
    label: {
        base: getCssVar('--graph-label-base', '#ffffff'),
        background: getCssVar('--graph-label-bg', 'rgba(0, 0, 0, 0.35)'),
        selected: getCssVar('--graph-label-selected', '#ffb3b3'),
        hover: getCssVar('--graph-label-hover', '#9fe7ff'),
    },
});

type EntityProvenanceArgs = {
    entityId: number;
    limit?: number;
};

type BeliefProvenanceArgs = {
    beliefId: number;
    kind: 'ics' | 'self';
    limit?: number;
};

const invokeWithArgs = <T, A extends Record<string, unknown>>(
    command: string,
    args: A,
    ms?: number,
) => invokeWithTimeout<T>(command, { args }, ms);

export function MemoryGraph3D({ isOpen, onClose }: Props) {
    const fgRef = useRef<any>();
    const materialsRef = useRef<Set<THREE.ShaderMaterial>>(new Set()); // Track materials for animation
    const rafRef = useRef<number>(); // Track animation frame
    const renderTimingStartRef = useRef<number | null>(null);
    const [graphData, setGraphData] = useState<MemoryGraph>({ nodes: [], links: [] });
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [editMode, setEditMode] = useState(false);
    const [editLabel, setEditLabel] = useState('');
    const [showAddPanel, setShowAddPanel] = useState(false);
    const [showEpisodicPanel, setShowEpisodicPanel] = useState(false);
    const [showClaimsPanel, setShowClaimsPanel] = useState(false);
    const [showScopePanel, setShowScopePanel] = useState(false);
    const [showRecallDebugPanel, setShowRecallDebugPanel] = useState(false);
    const [episodicEvents, setEpisodicEvents] = useState<EpisodicEvent[]>([]);
    const [episodicLoading, setEpisodicLoading] = useState(false);
    const [episodicError, setEpisodicError] = useState<string | null>(null);
    const [episodicQuery, setEpisodicQuery] = useState('');
    const [episodicType, setEpisodicType] = useState('');
    const [recallDebugQuery, setRecallDebugQuery] = useState('');
    const [recallDebugLoading, setRecallDebugLoading] = useState(false);
    const [recallDebugError, setRecallDebugError] = useState<string | null>(null);
    const [recallDebugResult, setRecallDebugResult] = useState<MemoryRetrievalDebugResponse | null>(null);
    const [claimOutcomes, setClaimOutcomes] = useState<ClaimOutcome[]>([]);
    const [claimsLoading, setClaimsLoading] = useState(false);
    const [claimsError, setClaimsError] = useState<string | null>(null);
    const [scopes, setScopes] = useState<string[]>([]);
    const [scopesLoading, setScopesLoading] = useState(false);
    const [scopesError, setScopesError] = useState<string | null>(null);
    const [dslInput, setDslInput] = useState('');
    const [writeResult, setWriteResult] = useState<{ success: boolean; message: string } | null>(null);
    const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
    const [provenanceEvents, setProvenanceEvents] = useState<EpisodicEvent[]>([]);
    const [provenanceLoading, setProvenanceLoading] = useState(false);
    const [provenanceError, setProvenanceError] = useState<string | null>(null);
    const [graphTheme, setGraphTheme] = useState(readGraphTheme);

    useEffect(() => {
        const handler = () => setGraphTheme(readGraphTheme());
        window.addEventListener(THEME_CHANGE_EVENT, handler);
        return () => window.removeEventListener(THEME_CHANGE_EVENT, handler);
    }, [graphTheme]);

    useEffect(() => {
        if (!isOpen) return;
        renderTimingStartRef.current = performance.now();
        const raf = requestAnimationFrame(() => {
            const start = renderTimingStartRef.current;
            if (start === null) return;
            const duration = performance.now() - start;
            renderTimingStartRef.current = null;
            void invokeWithTimeout("log_ui_timing", {
                event: "memory_graph_render",
                duration_ms: Math.round(duration),
            }).catch(() => {});
        });
        return () => cancelAnimationFrame(raf);
    }, [isOpen]);

    // Handle window resize
    useEffect(() => {
        if (!isOpen) return;
        const handleResize = () => {
            setDimensions({ width: window.innerWidth, height: window.innerHeight });
        };
        handleResize();
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [isOpen]);

    // Load graph
    const transformGraphData = useCallback((data: MemoryGraph): MemoryGraph => {
        const relationNodeIds = new Set<string>();
        const relationLabels = new Map<string, string>();
        const relationPolarities = new Map<string, string>();
        const relationParticipants = new Map<string, { role: string; entityId: string }[]>();

        for (const node of data.nodes) {
            if (node.nodeType === 'relation') {
                relationNodeIds.add(node.id);
                relationLabels.set(node.id, node.label);
                if (node.polarity) {
                    relationPolarities.set(node.id, node.polarity);
                }
            }
        }

        const retainedLinks: GraphLink[] = [];
        for (const link of data.links) {
            const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source;
            const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target;

            if (link.linkType === 'has_participant' && relationNodeIds.has(sourceId)) {
                const participants = relationParticipants.get(sourceId) ?? [];
                participants.push({ role: link.label || 'participant', entityId: String(targetId) });
                relationParticipants.set(sourceId, participants);
                continue;
            }

            if (relationNodeIds.has(sourceId) || relationNodeIds.has(targetId)) {
                continue;
            }

            retainedLinks.push({ ...link, source: sourceId, target: targetId });
        }

        const relationLinks: GraphLink[] = [];
        for (const [relId, participants] of relationParticipants.entries()) {
            if (participants.length < 2) {
                continue;
            }
            const relLabel = relationLabels.get(relId) || 'relation';
            const relPolarity = relationPolarities.get(relId);
            const denied = relPolarity === 'deny';
            const [first, second, ...rest] = participants;
            const pairs = rest.length === 0 ? [[first, second]] : rest.map((p) => [first, p]);
            for (const [left, right] of pairs) {
                const linkType = left.role === right.role
                    ? (denied ? 'relation_denied_bidirectional' : 'relation_bidirectional')
                    : (denied ? 'relation_denied' : 'relation');
                const label = denied ? `NOT ${relLabel}` : relLabel;
                relationLinks.push({
                    source: left.entityId,
                    target: right.entityId,
                    linkType,
                    label,
                    strength: 0.6,
                });
                if (left.role === right.role) {
                    relationLinks.push({
                        source: right.entityId,
                        target: left.entityId,
                        linkType,
                        label,
                        strength: 0.6,
                    });
                }
            }
        }

        const nodes = data.nodes.filter((node) => node.nodeType !== 'relation');
        return { nodes, links: [...retainedLinks, ...relationLinks] };
    }, []);

    const loadGraph = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await invokeWithTimeout<MemoryGraph>('memory_get_graph', { limit: 0 }, MEMORY_GRAPH_TIMEOUT_MS);
            const transformed = transformGraphData(data);
            const nodeIdSet = new Set(transformed.nodes.map((n) => n.id));
            const prunedLinks = transformed.links.filter((link) => {
                const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source;
                const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target;
                return nodeIdSet.has(sourceId) && nodeIdSet.has(targetId);
            });
            setGraphData({ ...transformed, links: prunedLinks });
        } catch (e: any) {
            setError(e.toString());
        } finally {
            setLoading(false);
        }
    }, [transformGraphData]);

    useEffect(() => {
        if (isOpen) loadGraph();
    }, [isOpen, loadGraph]);

    const sanitizedGraphData = useMemo(() => {
        const nodeIdSet = new Set(graphData.nodes.map((node) => node.id));
        const links = graphData.links.filter((link) => {
            const sourceId = typeof link.source === 'object' ? (link.source as any).id : link.source;
            const targetId = typeof link.target === 'object' ? (link.target as any).id : link.target;
            return nodeIdSet.has(sourceId) && nodeIdSet.has(targetId);
        });
        return { nodes: graphData.nodes, links };
    }, [graphData]);

    const formatEventSnippet = useCallback((event: EpisodicEvent) => {
        const payload = event.payload as { summary_snippet?: string };
        if (payload && typeof payload.summary_snippet === 'string' && payload.summary_snippet.trim().length > 0) {
            return payload.summary_snippet.trim();
        }
        return event.event_type;
    }, []);

    const formatEventTime = useCallback((timestamp: string) => {
        const parsed = Date.parse(timestamp);
        if (!Number.isNaN(parsed)) {
            return new Date(parsed).toLocaleString();
        }
        return timestamp;
    }, []);

    const copyToClipboard = useCallback(async (text: string, onError?: (message: string) => void) => {
        if (!text) return;
        try {
            await navigator.clipboard.writeText(text);
        } catch (err: any) {
            const message = err?.message || "Failed to copy to clipboard.";
            if (onError) {
                onError(message);
            }
        }
    }, []);

    const loadEpisodicEvents = useCallback(async () => {
        setEpisodicLoading(true);
        setEpisodicError(null);
        try {
            const trimmedQuery = episodicQuery.trim();
            const trimmedType = episodicType.trim();
            let events: EpisodicEvent[] = [];

            if (trimmedQuery || trimmedType) {
                events = await invokeWithTimeout<EpisodicEvent[]>('search_episodic_events', {
                    query: trimmedQuery || null,
                    event_type: trimmedType || null,
                    limit: EPISODIC_LIMIT
                }, MEMORY_GRAPH_TIMEOUT_MS);
            } else {
                events = await invokeWithTimeout<EpisodicEvent[]>('get_episodic_events', { limit: EPISODIC_LIMIT }, MEMORY_GRAPH_TIMEOUT_MS);
            }
            setEpisodicEvents(events);
        } catch (e: any) {
            setEpisodicError(e.toString());
        } finally {
            setEpisodicLoading(false);
        }
    }, [episodicQuery, episodicType]);

    const loadClaimOutcomes = useCallback(async () => {
        setClaimsLoading(true);
        setClaimsError(null);
        try {
            const outcomes = await invokeWithTimeout<ClaimOutcome[]>(
                "memory_get_claim_outcomes",
                { limit: 50 },
                MEMORY_GRAPH_TIMEOUT_MS
            );
            setClaimOutcomes(outcomes);
        } catch (e: any) {
            setClaimsError(e.toString());
        } finally {
            setClaimsLoading(false);
        }
    }, []);

    const loadScopes = useCallback(async () => {
        setScopesLoading(true);
        setScopesError(null);
        try {
            const data = await invokeWithTimeout<string[]>(
                "memory_get_scopes",
                { conversation_id: null },
                MEMORY_GRAPH_TIMEOUT_MS
            );
            setScopes(data);
        } catch (e: any) {
            setScopesError(e.toString());
        } finally {
            setScopesLoading(false);
        }
    }, []);

    const loadRecallDebug = useCallback(async () => {
        const query = recallDebugQuery.trim();
        if (!query) {
            setRecallDebugError("Enter a query to debug.");
            setRecallDebugResult(null);
            return;
        }
        setRecallDebugLoading(true);
        setRecallDebugError(null);
        try {
            const result = await invokeWithTimeout<MemoryRetrievalDebugResponse>(
                "memory_retrieval_debug",
                { query },
                MEMORY_GRAPH_TIMEOUT_MS,
            );
            setRecallDebugResult(result);
        } catch (err: any) {
            setRecallDebugError(err?.message || "Failed to load recall debug.");
            setRecallDebugResult(null);
        } finally {
            setRecallDebugLoading(false);
        }
    }, [recallDebugQuery]);

    const loadLastDebug = useCallback(async () => {
        if (recallDebugResult) {
            return;
        }
        try {
            const result = await invokeWithTimeout<MemoryRetrievalDebugResponse | null>(
                "memory_get_last_debug",
                {},
                MEMORY_GRAPH_TIMEOUT_MS
            );
            if (result) {
                setRecallDebugResult(result);
            }
        } catch {
            // ignore: last debug is optional
        }
    }, [recallDebugResult]);

    useEffect(() => {
        if (showEpisodicPanel && isOpen) {
            loadEpisodicEvents();
        }
    }, [showEpisodicPanel, loadEpisodicEvents, isOpen]);

    useEffect(() => {
        if (showClaimsPanel && isOpen) {
            loadClaimOutcomes();
        }
    }, [showClaimsPanel, loadClaimOutcomes, isOpen]);

    useEffect(() => {
        if (showScopePanel && isOpen) {
            loadScopes();
        }
    }, [showScopePanel, loadScopes, isOpen]);

    useEffect(() => {
        if (showRecallDebugPanel && isOpen) {
            loadLastDebug();
        }
    }, [showRecallDebugPanel, loadLastDebug, isOpen]);

    const parseNodeId = useCallback((nodeId: string) => {
        const parts = nodeId.split('_');
        if (parts.length < 2) return null;
        const parsed = Number(parts[1]);
        return Number.isFinite(parsed) ? parsed : null;
    }, []);

    useEffect(() => {
        let active = true;

        const loadProvenance = async () => {
            if (!isOpen || !selectedNode) {
                setProvenanceEvents([]);
                setProvenanceError(null);
                setProvenanceLoading(false);
                return;
            }

            const id = parseNodeId(selectedNode.id);
            if (!id) {
                setProvenanceEvents([]);
                setProvenanceError(null);
                setProvenanceLoading(false);
                return;
            }

            setProvenanceLoading(true);
            setProvenanceError(null);

            try {
                let events: EpisodicEvent[] = [];
                if (selectedNode.nodeType === 'entity') {
                    const args: EntityProvenanceArgs = { entityId: id, limit: PROVENANCE_LIMIT };
                    events = await invokeWithArgs<EpisodicEvent[], EntityProvenanceArgs>('memory_get_entity_provenance', args, MEMORY_GRAPH_TIMEOUT_MS);
                } else if (selectedNode.nodeType === 'fact' || selectedNode.nodeType === 'relation') {
                    const args: BeliefProvenanceArgs = { beliefId: id, kind: 'ics', limit: PROVENANCE_LIMIT };
                    events = await invokeWithArgs<EpisodicEvent[], BeliefProvenanceArgs>('memory_get_provenance', args, MEMORY_GRAPH_TIMEOUT_MS);
                }
                if (active) {
                    setProvenanceEvents(events);
                }
            } catch (e: any) {
                if (active) {
                    setProvenanceError(e.toString());
                }
            } finally {
                if (active) {
                    setProvenanceLoading(false);
                }
            }
        };

        loadProvenance();

        return () => {
            active = false;
        };
    }, [selectedNode, parseNodeId, isOpen]);

    // Bloom Post-Processing
    useEffect(() => {
        if (fgRef.current) {
            const composer = fgRef.current.postProcessingComposer(); // Bloom logic...

            // Resolution, Strength, Radius, Threshold
            const bloomPass = new UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                3.5, // Strength - Boosted for "Iron Man" intensity
                0.5, // Radius - Slightly wider glow
                0.1  // Threshold
            );
            composer.addPass(bloomPass);
        }
    }, [fgRef]);

    // Animation Loop
    useEffect(() => {
        if (!isOpen) {
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = undefined;
            }
            return;
        }

        const startTime = performance.now();

        const animate = () => {
            const time = (performance.now() - startTime) / 1000;

            // Update all tracked materials
            materialsRef.current.forEach(mat => {
                if (mat.uniforms && mat.uniforms.time) {
                    mat.uniforms.time.value = time;
                }
            });

            rafRef.current = requestAnimationFrame(animate);
        };

        rafRef.current = requestAnimationFrame(animate);

        return () => {
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = undefined;
            }
        };
    }, [isOpen]);

    // Node colors by type - rich gradients
    const getNodeColor = useCallback((node: GraphNode) => {
        if (node.nodeType === 'conflict') return graphTheme.node.conflict;
        if (node.nodeType === 'entity') {
            switch (node.entityType?.toLowerCase()) {
                case 'person': return graphTheme.node.person;
                case 'place': return graphTheme.node.place;
                case 'work': return graphTheme.node.work;
                case 'concept': return graphTheme.node.concept;
                case 'event': return graphTheme.node.event;
                case 'project': return graphTheme.node.project;
                case 'system': return graphTheme.node.system;
                default: return graphTheme.node.fallback;
            }
        }
        return graphTheme.node.muted;
    }, [graphTheme]);

    // Node size by importance
    const getNodeVal = useCallback((node: GraphNode) => {
        if (node.nodeType === 'fact') return 3;
        if (node.nodeType === 'conflict') return 8;
        const metric = node.activation !== undefined ? node.activation * 10 : Math.log((node.accessCount || 1) + 1);
        return Math.max(4, metric * 4);
    }, []);

    const getNodeBrightness = useCallback((node: GraphNode) => {
        const activation = node.activation ?? 0;
        const access = node.accessCount ?? 0;
        const accessNorm = Math.min(1, Math.log(access + 1) / Math.log(20));
        const strength = Math.max(activation, accessNorm);
        return 0.25 + 0.75 * strength;
    }, []);

    // Create glow texture for sprites
    const createGlowTexture = useCallback((color: string) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const context = canvas.getContext('2d');
        if (context) {
            const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
            // Core
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
            // Mid glow
            gradient.addColorStop(0.2, color);
            // Fade out
            gradient.addColorStop(0.5, 'rgba(0, 0, 0, 0)');

            context.fillStyle = gradient;
            context.fillRect(0, 0, 64, 64);
        }
        const texture = new THREE.CanvasTexture(canvas);
        return texture;
    }, []);

    // Generate texture once
    const techTexture = useMemo(() => createTechTexture(), []);

    // Create 3D sphere with glowing sprite halo
    const nodeThreeObject = useCallback((node: GraphNode) => {
        const baseColor = new THREE.Color(getNodeColor(node));
        const brightness = getNodeBrightness(node);
        baseColor.multiplyScalar(brightness);
        const colorHex = `#${baseColor.getHexString()}`;
        const size = node.nodeType === 'entity' ? 5 : 3;

        // Create group to hold sphere + glow + label
        const group = new THREE.Group();

        // 1. The Physical Sphere (Holographic)
        const geometry = new THREE.SphereGeometry(size, 32, 32);

        // Use custom hologram shader with texture
        const material = createHologramMaterial({ color: baseColor, map: techTexture });

        // Track for animation
        materialsRef.current.add(material);

        const sphere = new THREE.Mesh(geometry, material);
        group.add(sphere);

        // 2. The Glow Halo (Sprite) - OPTIONAL
        // Since the shader handles the glow, we might not need the sprite or we can reduce it.
        // Let's keep it but make it subtler to act as "volumetric atmosphere" around the core
        const glowTexture = createGlowTexture(colorHex);
        const spriteMaterial = new THREE.SpriteMaterial({
            map: glowTexture,
            transparent: true,
            opacity: 0.15 + 0.35 * brightness,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        // Scale sprite larger than sphere to create halo
        sprite.scale.set(size * 3, size * 3, 1); // Slightly smaller
        group.add(sprite);

        // 3. Text label floating above
        const label = new SpriteText(node.label);
        const labelBase = new THREE.Color(graphTheme.label.base).multiplyScalar(0.6 + 0.4 * brightness);
        label.color = `#${labelBase.getHexString()}`;
        label.textHeight = node.nodeType === 'entity' ? 4 : 3;
        label.backgroundColor = graphTheme.label.background;
        label.padding = 2;
        label.borderRadius = 4;
        (label as any).position.y = size + 8; // Bit higher
        group.add(label);

        return group;
    }, [getNodeColor, getNodeBrightness, createGlowTexture, graphTheme]);

    // Link styling - enhanced colors and opacity
    const getLinkColor = useCallback((link: GraphLink) => {
        if (link.linkType === 'has_fact') return rgbaFrom(graphTheme.link.fact, 0.25);
        if (link.linkType === 'belief_supersedes') return rgbaFrom(graphTheme.link.supersedes, 0.6);
        if (link.linkType === 'belief_contradicts') return rgbaFrom(graphTheme.link.contradicts, 0.95);
        if (link.linkType === 'belief_supports') return rgbaFrom(graphTheme.link.supports, 0.6);
        if (link.linkType === 'belief_derived_from') return rgbaFrom(graphTheme.link.derived, 0.5);
        if (link.linkType === 'in_conflict') return rgbaFrom(graphTheme.link.contradicts, 0.95);
        if (link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') return rgbaFrom(graphTheme.link.contradicts, 0.9);
        if (link.linkType === 'relation' || link.linkType === 'relation_bidirectional') return rgbaFrom(graphTheme.link.relation, 0.65);
        return rgbaFrom(graphTheme.link.other, 0.5);
    }, [graphTheme]);

    const getLinkWidth = useCallback((link: GraphLink) => {
        if (link.linkType === 'belief_contradicts' || link.linkType === 'in_conflict') return 3;
        if (link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') return 2.5;
        if (link.linkType === 'relation' || link.linkType === 'relation_bidirectional') return 2;
        if (link.linkType === 'has_fact') return 1;
        return 1.5;
    }, []);

    // Bright particle colors (full opacity)
    const getParticleColor = useCallback((link: GraphLink) => {
        // Return solid colors matching the types above, but without transparency
        if (link.linkType === 'has_fact') return graphTheme.link.fact;
        if (link.linkType === 'belief_supersedes') return graphTheme.link.supersedes;
        if (link.linkType === 'belief_contradicts') return graphTheme.link.contradicts;
        if (link.linkType === 'belief_supports') return graphTheme.link.supports;
        if (link.linkType === 'belief_derived_from') return graphTheme.link.derived;
        if (link.linkType === 'in_conflict') return graphTheme.link.contradicts;
        if (link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') return graphTheme.link.contradicts;
        if (link.linkType === 'relation' || link.linkType === 'relation_bidirectional') return graphTheme.link.relation;
        return graphTheme.link.other;
    }, [graphTheme]);

    const getParticleSpeed = useCallback((link: GraphLink) => {
        if (link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') return 0.012;
        if (link.linkType === 'relation' || link.linkType === 'relation_bidirectional') return 0.01;
        if (link.linkType === 'belief_contradicts' || link.linkType === 'in_conflict') return 0.015;
        return 0.005;
    }, []);

    const getRelationLabel = useCallback((link: GraphLink) => {
        if (link.linkType === 'relation' || link.linkType === 'relation_bidirectional'
            || link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') {
            return link.label || '';
        }
        return '';
    }, []);

    const linkThreeObject = useCallback((link: GraphLink) => {
        if (link.linkType !== 'relation' && link.linkType !== 'relation_bidirectional'
            && link.linkType !== 'relation_denied' && link.linkType !== 'relation_denied_bidirectional') {
            const hidden = new THREE.Object3D();
            hidden.visible = false;
            return hidden;
        }
        const label = new SpriteText(link.label || '');
        if (link.linkType === 'relation_denied' || link.linkType === 'relation_denied_bidirectional') {
            label.color = graphTheme.label.selected;
        } else {
            label.color = graphTheme.label.hover;
        }
        label.textHeight = 2.5;
        label.backgroundColor = graphTheme.label.background;
        label.padding = 1;
        label.borderRadius = 3;
        return label;
    }, [graphTheme]);

    const linkPositionUpdate = useCallback((sprite: THREE.Object3D, coords: any) => {
        if (!sprite || !coords) return;
        const { start, end } = coords;
        sprite.position.x = (start.x + end.x) / 2;
        sprite.position.y = (start.y + end.y) / 2;
        sprite.position.z = (start.z + end.z) / 2;
    }, []);

    // Click handler
    const handleNodeClick = useCallback((node: GraphNode) => {
        setSelectedNode(node);
        setEditLabel(node.label);
        setEditMode(false);
        setContextMenu(null);

        // Focus camera
        const distance = 100;
        const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);
        fgRef.current?.cameraPosition(
            { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
            node,
            1000
        );
    }, []);

    const focusNodeByBelief = useCallback((beliefId: number) => {
        const factNode = graphData.nodes.find((node) => node.id === `fact_${beliefId}`);
        if (factNode) {
            handleNodeClick(factNode);
            return;
        }
        const relNode = graphData.nodes.find((node) => node.id === `rel_${beliefId}`);
        if (relNode) {
            handleNodeClick(relNode);
        }
    }, [graphData.nodes, handleNodeClick]);

    // Right-click handler
    const handleNodeRightClick = useCallback((node: GraphNode, event: MouseEvent) => {
        setContextMenu({ x: event.clientX, y: event.clientY, node });
    }, []);

    // Search and focus
    const handleSearch = useCallback(() => {
        if (!searchQuery.trim()) return;
        const node = graphData.nodes.find(n =>
            n.label.toLowerCase().includes(searchQuery.toLowerCase())
        );
        if (node) {
            handleNodeClick(node);
        }
    }, [searchQuery, graphData.nodes, handleNodeClick]);

    // Update entity
    const handleSave = useCallback(async () => {
        if (!selectedNode || selectedNode.nodeType !== 'entity') return;
        setError(null);
        const entityId = parseInt(selectedNode.id.split('_')[1]);
        try {
            await invokeWithTimeout('memory_update_entity', { entityId, newLabel: editLabel }, MEMORY_GRAPH_TIMEOUT_MS);
        } catch (e: any) {
            setError(e.toString());
            return;
        }

        // Update local state
        setGraphData(prev => ({
            ...prev,
            nodes: prev.nodes.map(n =>
                n.id === selectedNode.id ? { ...n, label: editLabel } : n
            )
        }));
        setSelectedNode(prev => prev ? { ...prev, label: editLabel } : null);
        setEditMode(false);
    }, [selectedNode, editLabel]);

    // Delete node
    const handleDelete = useCallback(async (node: GraphNode) => {
        const confirmed = confirm(`Delete "${node.label}"? This cannot be undone.`);
        if (!confirmed) return;

        const id = parseInt(node.id.split('_')[1]);
        try {
            if (node.nodeType === 'entity') {
                await invokeWithTimeout('memory_delete_entity', { entityId: id }, MEMORY_GRAPH_TIMEOUT_MS);
            } else {
                await invokeWithTimeout('memory_delete_belief', { beliefId: id }, MEMORY_GRAPH_TIMEOUT_MS);
            }
            await loadGraph();
            setSelectedNode(null);
            setContextMenu(null);
        } catch (e: any) {
            setError(e.toString());
        }
    }, [loadGraph]);

    // Write DSL to memory
    const handleWrite = useCallback(async () => {
        if (!dslInput.trim()) return;
        setWriteResult(null);
        try {
            const result = await invokeWithTimeout<CompileResult>('memory_write', {
                input: dslInput,
                source: 'user',
                admin_override: true
            }, MEMORY_WRITE_TIMEOUT_MS);
            if (result.errors.length > 0) {
                setWriteResult({ success: false, message: result.errors.join(', ') });
            } else {
                setWriteResult({ success: true, message: `Written ${result.written_ids.length} items` });
                setDslInput('');
                // Reload graph to show new nodes
                await loadGraph();
            }
        } catch (e: any) {
            setWriteResult({ success: false, message: e.toString() });
        }
    }, [dslInput, loadGraph]);

    // Close context menu on click outside
    useEffect(() => {
        const handler = () => setContextMenu(null);
        if (contextMenu) {
            document.addEventListener('click', handler);
            return () => document.removeEventListener('click', handler);
        }
    }, [contextMenu]);

    useEffect(() => {
        if (!isOpen) return;
        const handler = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="memory-graph-overlay">
            <div className="memory-graph-container">
                {/* Header */}
                <div className="memory-graph-header">
                    <h2>Memory Graph</h2>
                    <div className="memory-graph-search">
                        <input
                            type="text"
                            className="input"
                            placeholder="Search..."
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleSearch()}
                        />
                        <button
                            className="btn btn-primary"
                            onClick={handleSearch}
                            aria-label="Search memory graph"
                        >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="11" cy="11" r="7" />
                                <line x1="16.65" y1="16.65" x2="21" y2="21" />
                            </svg>
                        </button>
                        <button
                            className={`btn ${showAddPanel ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setShowAddPanel(!showAddPanel)}
                            title="Add Node"
                        >
                            +
                        </button>
                        <button
                            className={`btn ${showEpisodicPanel ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setShowEpisodicPanel(!showEpisodicPanel)}
                            title="Episodic Memory"
                        >
                            Episodic
                        </button>
                        <button
                            className={`btn ${showClaimsPanel ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setShowClaimsPanel(!showClaimsPanel)}
                            title="Claim Outcomes"
                        >
                            Claims
                        </button>
                        <button
                            className={`btn ${showScopePanel ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setShowScopePanel(!showScopePanel)}
                            title="Scopes"
                        >
                            Scope
                        </button>
                        <button
                            className={`btn ${showRecallDebugPanel ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setShowRecallDebugPanel(!showRecallDebugPanel)}
                            title="Memory Debug"
                        >
                            Debug
                        </button>
                    </div>
                    <button className="btn btn-secondary close-btn" onClick={onClose} aria-label="Close memory graph">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    </button>
                </div>

                {/* Add Node Panel */}
                {showAddPanel && (
                    <div className="add-node-panel">
                        <h4 className="add-node-title">Add to Memory</h4>
                        <textarea
                            placeholder={`Enter DSL statements...\n$user:favorite_color = "blue" ^today @global\nparent_of(parent: #Mister Black -> child: #Hannah)\nworks_with(person: $user, person: #Jane Doe) @project:alpha`}
                            value={dslInput}
                            onChange={e => setDslInput(e.target.value)}
                            className="add-node-textarea"
                        />
                        <div className="add-node-actions">
                            <button className="btn btn-primary btn-flex" onClick={handleWrite}>
                                Write to Memory
                            </button>
                            <button className="btn btn-secondary" onClick={() => setShowAddPanel(false)}>
                                Close
                            </button>
                        </div>
                        {writeResult && (
                            <div className={`add-node-result ${writeResult.success ? "success" : "error"}`}>
                                {writeResult.message}
                            </div>
                        )}
                    </div>
                )}

                {showEpisodicPanel && (
                    <div className="episodic-panel">
                        <div className="episodic-panel-header">
                            <h4>Episodic Memory</h4>
                            <button className="btn btn-secondary" onClick={loadEpisodicEvents}>
                                Refresh
                            </button>
                        </div>
                        <div className="episodic-panel-controls">
                            <input
                                className="input"
                                placeholder="Search events..."
                                value={episodicQuery}
                                onChange={(e) => setEpisodicQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && loadEpisodicEvents()}
                            />
                            <input
                                className="input"
                                placeholder="Event type (optional)"
                                value={episodicType}
                                onChange={(e) => setEpisodicType(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && loadEpisodicEvents()}
                            />
                            <button className="btn btn-primary" onClick={loadEpisodicEvents}>
                                Search
                            </button>
                        </div>
                        <div className="episodic-panel-list">
                            {episodicLoading && <div className="episodic-panel-empty">Loading episodic events...</div>}
                            {!episodicLoading && episodicError && (
                                <div className="episodic-panel-error">{episodicError}</div>
                            )}
                            {!episodicLoading && !episodicError && episodicEvents.length === 0 && (
                                <div className="episodic-panel-empty">No episodic events found.</div>
                            )}
                            {!episodicLoading && !episodicError && episodicEvents.map((event) => (
                                <div key={event.id} className="episodic-panel-item">
                                    <div className="episodic-panel-item-header">
                                        <span className="episodic-panel-type">{event.event_type}</span>
                                        <span className="episodic-panel-time">{formatEventTime(event.timestamp)}</span>
                                    </div>
                                    <div className="episodic-panel-snippet">{formatEventSnippet(event)}</div>
                                    {event.linked_belief_id && (
                                        <div className="episodic-panel-meta">Belief ID: {event.linked_belief_id}</div>
                                    )}
                                    {event.linked_belief_id && (
                                        <button
                                            className="btn btn-secondary episodic-panel-focus"
                                            onClick={() => focusNodeByBelief(event.linked_belief_id as number)}
                                        >
                                            Focus
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {showClaimsPanel && (
                    <div className="episodic-panel">
                        <div className="episodic-panel-header">
                            <h4>Claims</h4>
                            <button className="btn btn-secondary" onClick={loadClaimOutcomes}>
                                Refresh
                            </button>
                        </div>
                        <div className="episodic-panel-list">
                            {claimsLoading && <div className="episodic-panel-empty">Loading claim outcomes...</div>}
                            {!claimsLoading && claimsError && (
                                <div className="episodic-panel-error">{claimsError}</div>
                            )}
                            {!claimsLoading && !claimsError && claimOutcomes.length === 0 && (
                                <div className="episodic-panel-empty">No claim outcomes recorded.</div>
                            )}
                            {!claimsLoading && !claimsError && claimOutcomes.slice().reverse().map((outcome) => (
                                <div key={`${outcome.claim_id}-${outcome.created_at}`} className="episodic-panel-item">
                                    <div className="episodic-panel-item-header">
                                        <span className="episodic-panel-type">{outcome.status}</span>
                                        <span className="episodic-panel-time">{formatEventTime(outcome.created_at)}</span>
                                    </div>
                                    <div className="episodic-panel-snippet">
                                        {outcome.reason || "No reason provided."}
                                    </div>
                                    <div className="episodic-panel-meta">Claim ID: {outcome.claim_id}</div>
                                    <div className="episodic-panel-meta">Scope: {outcome.scope}</div>
                                    {outcome.session_id && (
                                        <div className="episodic-panel-meta">Session: {outcome.session_id}</div>
                                    )}
                                    <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => copyToClipboard(outcome.claim_id, (msg) => setClaimsError(msg))}
                                        >
                                            Copy ID
                                        </button>
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => copyToClipboard(outcome.reason || '', (msg) => setClaimsError(msg))}
                                            disabled={!outcome.reason}
                                        >
                                            Copy Reason
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {showScopePanel && (
                    <div className="episodic-panel">
                        <div className="episodic-panel-header">
                            <h4>Scope</h4>
                            <button className="btn btn-secondary" onClick={loadScopes}>
                                Refresh
                            </button>
                        </div>
                        <div className="episodic-panel-list">
                            {scopesLoading && <div className="episodic-panel-empty">Loading scopes...</div>}
                            {!scopesLoading && scopesError && (
                                <div className="episodic-panel-error">{scopesError}</div>
                            )}
                            {!scopesLoading && !scopesError && scopes.length === 0 && (
                                <div className="episodic-panel-empty">No active scopes.</div>
                            )}
                            {!scopesLoading && !scopesError && scopes.length > 0 && (
                                <div className="episodic-panel-item">
                                    <div className="episodic-panel-item-header">
                                        <span className="episodic-panel-type">Active Scopes</span>
                                    </div>
                                    <div className="episodic-panel-snippet">{scopes.join(', ')}</div>
                                    {recallDebugResult?.log.retrieval ? (
                                        <>
                                            <div className="episodic-panel-meta">
                                                Shadowed (last debug): {recallDebugResult.log.retrieval.shadowing?.beliefs_shadowed ?? 0}
                                            </div>
                                            <div className="episodic-panel-meta">
                                                Topic groups: {recallDebugResult.log.retrieval.shadowing?.topic_groups_count ?? 0}
                                            </div>
                                        </>
                                    ) : (
                                        <div className="episodic-panel-meta">Run Debug to see shadowing counts.</div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {showRecallDebugPanel && (
                    <div className="episodic-panel">
                        <div className="episodic-panel-header">
                            <h4>Debug</h4>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button className="btn btn-secondary" onClick={loadRecallDebug}>
                                    Run Debug
                                </button>
                                <button
                                    className="btn btn-secondary"
                                    onClick={() => copyToClipboard(JSON.stringify(recallDebugResult?.log || {}, null, 2), (msg) => setRecallDebugError(msg))}
                                    disabled={!recallDebugResult}
                                >
                                    Copy JSON
                                </button>
                            </div>
                        </div>
                        <div className="episodic-panel-controls">
                            <input
                                className="input"
                                placeholder="Enter a query to debug..."
                                value={recallDebugQuery}
                                onChange={(e) => setRecallDebugQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && loadRecallDebug()}
                            />
                        </div>
                        <div className="episodic-panel-list">
                            {recallDebugLoading && <div className="episodic-panel-empty">Running recall debug...</div>}
                            {!recallDebugLoading && recallDebugError && (
                                <div className="episodic-panel-error">{recallDebugError}</div>
                            )}
                            {!recallDebugLoading && !recallDebugError && !recallDebugResult && (
                                <div className="episodic-panel-empty">No debug output yet.</div>
                            )}
                            {!recallDebugLoading && !recallDebugError && recallDebugResult && (
                                <div className="episodic-panel-item">
                                    <div className="episodic-panel-item-header">
                                        <span className="episodic-panel-type">Summary</span>
                                    </div>
                                    <div className="episodic-panel-snippet">
                                        {recallDebugResult.summary || "No summary."}
                                    </div>
                                    <div className="episodic-panel-meta">Timestamp: {recallDebugResult.log.timestamp}</div>
                                    <div className="episodic-panel-meta">Anchors: {recallDebugResult.log.retrieval?.anchors?.length ?? 0}</div>
                                    <div className="episodic-panel-meta">Facts: {recallDebugResult.log.retrieval?.selection?.facts_final ?? 0}</div>
                                    <div className="episodic-panel-meta">Duration: {recallDebugResult.log.retrieval?.duration_ms ?? 0} ms</div>
                                    <pre className="recall-debug-pre">
                                        {JSON.stringify(recallDebugResult.log, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Graph */}
                <div className="memory-graph-canvas" onContextMenu={e => e.preventDefault()}>
                    {loading && <div className="loading">Loading graph...</div>}
                    {error && (
                        <div className="error">
                            <div>{error}</div>
                            <button className="btn btn-secondary" onClick={loadGraph} style={{ marginTop: '12px' }}>
                                Retry
                            </button>
                        </div>
                    )}

                    {!loading && !error && sanitizedGraphData.nodes.length > 0 && (
                        <ForceGraph3D
                            ref={fgRef}
                            width={dimensions.width}
                            height={dimensions.height - 80} // Account for header
                            graphData={sanitizedGraphData}
                            nodeThreeObject={nodeThreeObject}
                            nodeThreeObjectExtend={false}
                            nodeVal={getNodeVal}
                            linkColor={getLinkColor}
                            linkWidth={getLinkWidth}
                            linkOpacity={0.15} // Very subtle/transparent track
                            linkLabel={getRelationLabel}
                            linkThreeObject={linkThreeObject}
                            linkThreeObjectExtend={true}
                            linkPositionUpdate={linkPositionUpdate}
                            onNodeClick={handleNodeClick}
                            onNodeRightClick={handleNodeRightClick}
                            backgroundColor="#00000000" /* Transparent so CSS bg shows through */
                            showNavInfo={false}
                            linkDirectionalParticles={2}
                            linkDirectionalParticleSpeed={getParticleSpeed}
                            linkDirectionalParticleWidth={2} // Reduced back to normal size
                            linkDirectionalParticleColor={getParticleColor} // Keep solid colors
                            d3AlphaDecay={0.015}
                            d3VelocityDecay={0.25}
                            warmupTicks={100}
                            cooldownTicks={200}
                            onEngineStop={() => fgRef.current?.zoomToFit(400, 80)}
                        />
                    )}
                    {!loading && !error && sanitizedGraphData.nodes.length === 0 && (
                        <div className="loading">No graph nodes to display.</div>
                    )}

                    {selectedNode && (
                        <div className="node-detail-panel">
                            <div className="panel-header">
                                <span className={`type-badge ${selectedNode.nodeType}`}>
                                    {selectedNode.entityType || selectedNode.nodeType}
                                </span>
                                <button
                                    className="btn btn-secondary btn-compact"
                                    onClick={() => setSelectedNode(null)}
                                    aria-label="Close details"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            </div>

                            <div className="panel-body">
                                {editMode ? (
                                    <input
                                        type="text"
                                        className="input"
                                        value={editLabel}
                                        onChange={e => setEditLabel(e.target.value)}
                                        onKeyDown={e => e.key === 'Enter' && handleSave()}
                                        autoFocus
                                    />
                                ) : (
                                    <h3>{selectedNode.label}</h3>
                                )}

                                {selectedNode.confidence !== undefined && selectedNode.confidence !== null && (
                                    <div className="confidence">
                                        <span>Confidence:</span>
                                        <div className="confidence-bar">
                                            <div style={{ width: `${selectedNode.confidence * 100}%` }}></div>
                                        </div>
                                        <span>{(selectedNode.confidence * 100).toFixed(0)}%</span>
                                    </div>
                                )}

                                {selectedNode.key && (
                                    <p className="fact-detail">
                                        <strong>{selectedNode.key}</strong> {selectedNode.value}
                                    </p>
                                )}

                                {selectedNode.activation !== undefined && selectedNode.activation > 0 && (
                                    <div className="node-meta node-meta-spaced">
                                        Activation: {(selectedNode.activation * 100).toFixed(1)}%
                                    </div>
                                )}
                                <div className="node-meta">
                                    Access Count: {selectedNode.accessCount}
                                </div>

                                {/* Temporal and Scope Info */}
                                {(selectedNode.scope || selectedNode.polarity || selectedNode.timeBucketKind) && (
                                    <div className="temporal-scope-info">
                                        {selectedNode.scope && (
                                            <span className={`scope-badge ${selectedNode.scope.toLowerCase()}`}>
                                                {selectedNode.scope}
                                            </span>
                                        )}
                                        {selectedNode.polarity && (
                                            <span className={`polarity-badge ${selectedNode.polarity.toLowerCase()}`}>
                                                {selectedNode.polarity === 'positive' ? 'POS' : selectedNode.polarity === 'negative' ? 'NEG' : '~'}
                                            </span>
                                        )}
                                        {selectedNode.timeBucketKind && selectedNode.timeBucketValue && (
                                            <span className="time-badge">
                                                Time: {selectedNode.timeBucketKind}: {selectedNode.timeBucketValue}
                                            </span>
                                        )}
                                    </div>
                                )}

                                <div className="provenance-section">
                                    <div className="provenance-title">Provenance</div>
                                    {provenanceLoading && (
                                        <div className="provenance-empty">Loading episodic links...</div>
                                    )}
                                    {!provenanceLoading && provenanceError && (
                                        <div className="provenance-error">{provenanceError}</div>
                                    )}
                                    {!provenanceLoading && !provenanceError && provenanceEvents.length === 0 && (
                                        <div className="provenance-empty">No episodic links yet.</div>
                                    )}
                                    {!provenanceLoading && !provenanceError && provenanceEvents.length > 0 && (
                                        <div className="provenance-list">
                                            {provenanceEvents.map((event) => (
                                                <div key={event.id} className="provenance-item">
                                                    <div className="provenance-item-header">
                                                        <span className="provenance-type">{event.event_type}</span>
                                                        <span className="provenance-time">{formatEventTime(event.timestamp)}</span>
                                                    </div>
                                                    <div className="provenance-snippet">{formatEventSnippet(event)}</div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="panel-actions">
                                {editMode ? (
                                    <>
                                        <button className="btn btn-primary" onClick={handleSave}>Save</button>
                                        <button className="btn btn-secondary" onClick={() => setEditMode(false)}>Cancel</button>
                                    </>
                                ) : (
                                    <>
                                        {selectedNode.nodeType === 'entity' && (
                                            <button className="btn btn-primary" onClick={() => setEditMode(true)}>Edit</button>
                                        )}
                                        <button className="btn btn-danger-outline" onClick={() => handleDelete(selectedNode)}>Delete</button>
                                    </>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Legend */}
                <div className="memory-graph-legend">
                    <span className="legend-item"><span className="dot legend-dot person"></span> Person</span>
                    <span className="legend-item"><span className="dot legend-dot place"></span> Place</span>
                    <span className="legend-item"><span className="dot legend-dot work"></span> Work</span>
                    <span className="legend-item"><span className="dot legend-dot concept"></span> Concept</span>
                    <span className="legend-item"><span className="dot legend-dot event"></span> Event</span>
                    <span className="legend-item"><span className="dot legend-dot project"></span> Project</span>
                    <span className="legend-item"><span className="dot legend-dot system"></span> System</span>
                    <span className="legend-item"><span className="dot legend-dot conflict"></span> Conflict</span>
                </div>

                {/* Context Menu */}
                {contextMenu && (
                    <div
                        className="context-menu"
                        style={{ left: contextMenu.x, top: contextMenu.y }}
                        onClick={e => e.stopPropagation()}
                    >
                        <div className="context-menu-item" onClick={() => {
                            setSelectedNode(contextMenu.node);
                            setEditLabel(contextMenu.node.label);
                            setEditMode(true);
                            setContextMenu(null);
                        }}>
                            Edit
                        </div>
                        <div className="context-menu-item" onClick={() => handleNodeClick(contextMenu.node)}>
                            Details
                        </div>
                        <div className="context-menu-item danger" onClick={() => handleDelete(contextMenu.node)}>
                            Delete
                        </div>
                    </div>
                )}

                {/* Stats */}
                <div className="memory-graph-stats">
                    {graphData.nodes.length} nodes | {graphData.links.length} links
                </div>
            </div>
        </div>
    );
}

export default MemoryGraph3D;

