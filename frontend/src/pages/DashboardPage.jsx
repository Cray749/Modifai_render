import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Plus,
    ArrowRight,
    FileText,
    Database,
    Target,
    Brain,
    Rocket,
    Activity,
    Box,
    Loader2,
    ChevronRight,
    Server,
    Layers,
    Bot,
    Zap
} from 'lucide-react';
import { apiClient } from '@/api/client';
import BorderGlow from '@/components/ui/BorderGlow';
import SpotlightCard from '@/components/ui/SpotlightCard';

export default function DashboardPage() {
    const navigate = useNavigate();
    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(true);
    const [timelineLogs, setTimelineLogs] = useState([]);

    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const data = await apiClient.get('projects/');
                const mapped = data.map(p => ({
                    id: p.id,
                    name: p.name,
                    description: p.description || '',
                    status: p.status,
                    createdAt: p.created_at,
                    model: p.base_model || 'Unknown',
                    mode: p.mode,
                    filesCount: 0,
                    currentStep: 0,
                    pipeline: [],
                }));
                setProjects(mapped);

                // Fetch logs for the most recent project to populate timeline
                if (mapped.length > 0) {
                    const latestProj = mapped[0];
                    try {
                        const logsData = await apiClient.get(`projects/${latestProj.id}/logs`);
                        const logs = logsData.logs || [];
                        
                        // Format for ActivityTimeline: { id, event, time, status }
                        const formatted = logs
                            .filter(log => !['PassStateEntered', 'PassStateExited', 'WaitStateEntered', 'WaitStateExited'].includes(log.type))
                            .map((log, idx) => {
                                let status = 'success';
                                if (log.type.includes('Failed') || log.type.includes('Aborted') || log.type.includes('TimedOut')) status = 'error';
                                else if (log.type.includes('Started') || log.type.includes('Entered')) status = 'running';

                                const timeStr = new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                                const cleanLabel = log.label.replace('Entered: ', 'Starting ').replace('Exited: ', 'Finished ').replace(/_/g, ' ');

                                return {
                                    id: log.id || String(idx),
                                    event: cleanLabel,
                                    time: timeStr,
                                    status: status
                                };
                            })
                            .slice(0, 5); // top 5 most recent

                        setTimelineLogs(formatted);
                    } catch (logErr) {
                        console.error('Failed to fetch timeline logs:', logErr);
                    }
                }
            } catch (err) {
                console.error('Failed to fetch projects for dashboard:', err);
            } finally {
                setLoading(false);
            }
        };
        fetchProjects();
    }, []);

    const stats = {
        totalProjects: projects.length,
        modelsDeployed: projects.filter(p => p.mode === 'full' && p.status === 'completed').length,
        datasetsGenerated: projects.filter(p => p.status === 'completed').length,
        running: projects.filter(p => p.status === 'running').length,
    };

    const recentProjects = projects.slice(0, 5);

    const getStatusColor = (status) => {
        switch (status.toLowerCase()) {
            case 'completed': return 'bg-[#0070f3]'; // success / link
            case 'running': return 'bg-[#f5a623]'; // warning
            case 'failed': return 'bg-[#ee0000]'; // error
            default: return 'bg-[#a1a1a1]'; // pending / hairline-strong
        }
    };

    return (
        <div className="min-h-screen bg-canvas-soft text-ink font-sans selection:bg-ink selection:text-canvas">
            {/* Embedded styles for organic backdrop drifting */}
            <style>{`
                @keyframes blob-drift-1 {
                    0% { 
                        transform: translate(0px, 0px) scale(1); 
                        opacity: calc(var(--blob-base-opacity) * 0.85); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                    33% { 
                        transform: translate(260px, -180px) scale(1.15); 
                        opacity: calc(var(--blob-base-opacity) * 1); 
                        filter: blur(calc(var(--blob-base-blur) * 1.15)); 
                    }
                    66% { 
                        transform: translate(-180px, 140px) scale(0.85); 
                        opacity: calc(var(--blob-base-opacity) * 0.75); 
                        filter: blur(calc(var(--blob-base-blur) * 0.85)); 
                    }
                    100% { 
                        transform: translate(0px, 0px) scale(1); 
                        opacity: calc(var(--blob-base-opacity) * 0.85); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                }
                @keyframes blob-drift-2 {
                    0% { 
                        transform: translate(0px, 0px) scale(1); 
                        opacity: calc(var(--blob-base-opacity) * 0.8); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                    50% { 
                        transform: translate(-280px, 200px) scale(0.85); 
                        opacity: calc(var(--blob-base-opacity) * 1); 
                        filter: blur(calc(var(--blob-base-blur) * 0.9)); 
                    }
                    100% { 
                        transform: translate(0px, 0px) scale(1); 
                        opacity: calc(var(--blob-base-opacity) * 0.8); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                }
                @keyframes blob-drift-3 {
                    0% { 
                        transform: translate(0px, 0px) scale(0.85); 
                        opacity: calc(var(--blob-base-opacity) * 0.9); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                    50% { 
                        transform: translate(220px, 260px) scale(1.15); 
                        opacity: calc(var(--blob-base-opacity) * 1); 
                        filter: blur(calc(var(--blob-base-blur) * 1.1)); 
                    }
                    100% { 
                        transform: translate(0px, 0px) scale(0.85); 
                        opacity: calc(var(--blob-base-opacity) * 0.9); 
                        filter: blur(var(--blob-base-blur)); 
                    }
                }
                .animate-blob-1 {
                    --blob-base-opacity: 1;
                    --blob-base-blur: 100px;
                    filter: blur(var(--blob-base-blur));
                    opacity: var(--blob-base-opacity);
                    animation: blob-drift-1 13.8s ease-in-out infinite;
                }
                .animate-blob-2 {
                    --blob-base-opacity: 1;
                    --blob-base-blur: 100px;
                    filter: blur(var(--blob-base-blur));
                    opacity: var(--blob-base-opacity);
                    animation: blob-drift-2 16.6s ease-in-out infinite;
                }
                .animate-blob-3 {
                    --blob-base-opacity: 0.7;
                    --blob-base-blur: 120px;
                    filter: blur(var(--blob-base-blur));
                    opacity: var(--blob-base-opacity);
                    animation: blob-drift-3 20.2s ease-in-out infinite;
                }
                .dark .animate-blob-1 {
                    --blob-base-opacity: 0.44;
                }
                .dark .animate-blob-2 {
                    --blob-base-opacity: 0.44;
                }
                .dark .animate-blob-3 {
                    --blob-base-opacity: 0.33;
                }
            `}</style>

            {/* Mesh gradient atmospheric backdrop - Hero Scale only */}
            <div className="absolute top-0 left-0 right-0 h-[700px] overflow-hidden pointer-events-none z-0">
                {/* Blob 1: Left Cyan/Blue */}
                <div className="absolute top-[-25%] left-[-15%] w-[65%] h-[130%] rounded-full bg-gradient-to-tr from-[#007cf0]/25 to-[#00dfd8]/20 animate-blob-1" />
                
                {/* Blob 2: Right Purple/Pink */}
                <div className="absolute top-[-15%] right-[-15%] w-[55%] h-[110%] rounded-full bg-gradient-to-tr from-[#7928ca]/25 to-[#ff0080]/20 animate-blob-2" />
                
                {/* Blob 3: Center Subtle Overlay */}
                <div className="absolute top-[-20%] left-[20%] w-[45%] h-[90%] rounded-full bg-gradient-to-tr from-[#007cf0]/15 to-[#7928ca]/15 animate-blob-3" />
                
                {/* Smooth Fade Overlay to prevent abrupt cutoff */}
                <div 
                    className="absolute bottom-0 left-0 right-0 h-[250px] pointer-events-none"
                    style={{
                        background: 'linear-gradient(to top, var(--canvas-soft) 0%, transparent 100%)'
                    }}
                />
            </div>

            <div className="relative z-10 max-w-[1400px] mx-auto px-4 md:px-6 py-16 md:py-24 space-y-24 md:space-y-32">

                {/* ── Section 1: Hero Command Center ── */}
                <section className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-16">
                    <div className="max-w-2xl space-y-6">
                        <div className="font-mono text-[12px] leading-[16px] uppercase text-mute tracking-widest">
                            Model Factory
                        </div>
                        <h1 className="text-[40px] md:text-[48px] leading-[44px] md:leading-[48px] font-semibold tracking-[-2.4px] text-ink">
                            Build specialized AI systems from raw knowledge.
                        </h1>
                        <p className="text-[18px] leading-[28px] text-body">
                            Transform documents into synthetic datasets, fine-tune models, and deploy as a managed API endpoint.
                        </p>
                        <div className="flex items-center gap-4 pt-4">
                            <button 
                                onClick={() => navigate('/projects/new')}
                                className="bg-ink text-canvas text-[16px] leading-[24px] font-medium h-[48px] px-6 rounded-full hover:bg-ink/90 transition-colors flex items-center gap-2 shadow-sm"
                            >
                                <Plus className="w-5 h-5" />
                                New Project
                            </button>
                            <button 
                                onClick={() => navigate('/projects')}
                                className="bg-canvas text-ink text-[16px] leading-[24px] font-medium h-[48px] px-6 rounded-full border border-hairline hover:bg-canvas-soft transition-colors flex items-center gap-2 shadow-[0px_1px_1px_#00000005,0px_2px_2px_#0000000a]"
                            >
                                View Projects
                                <ArrowRight className="w-4 h-4 text-mute" />
                            </button>
                        </div>
                    </div>

                    {/* Animated System Diagram */}
                    <PipelineDiagram />
                </section>

                {/* ── Section 2: Operational Overview ── */}
                <section>
                    <h2 className="font-mono text-[12px] leading-[16px] uppercase text-mute tracking-widest mb-6">
                        Operational Overview
                    </h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {[
                            { label: 'Projects', value: stats.totalProjects, icon: Box },
                            { label: 'Active Pipelines', value: stats.running, icon: Activity },
                            { label: 'Datasets', value: stats.datasetsGenerated, icon: Database },
                            { label: 'Deployments', value: stats.modelsDeployed, icon: Server }
                        ].map((stat, i) => (
                            <BorderGlow
                                key={i}
                                borderRadius={8}
                                backgroundColor="var(--canvas)"
                                edgeSensitivity={30}
                                glowColor="40 80 80"
                                colors={['#c084fc', '#f472b6', '#38bdf8']}
                                animated={false}
                                glowRadius={40}
                                glowIntensity={1.0}
                            >
                                <div className="p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <span className="text-[14px] leading-[20px] font-medium text-body">{stat.label}</span>
                                        <stat.icon className="w-4 h-4 text-mute" strokeWidth={1.5} />
                                    </div>
                                    <div className="flex items-end justify-between">
                                        <span className="text-[32px] leading-[40px] font-semibold tracking-[-1.28px] text-ink">
                                            {stat.value}
                                        </span>
                                        {stat.label === 'Active Pipelines' && stat.value > 0 && (
                                            <div className="flex items-center gap-1.5 mb-2">
                                                <div className="w-2 h-2 rounded-full bg-[#0070f3] animate-pulse" />
                                                <span className="text-[12px] leading-[16px] text-[#0070f3] font-medium">Running</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </BorderGlow>
                        ))}
                    </div>
                </section>

                {/* ── Grid for Section 3 (Table) & Section 5 (Timeline) ── */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                    {/* ── Section 3: Active Workbench ── */}
                    <section className="lg:col-span-2">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="font-mono text-[12px] leading-[16px] uppercase text-mute tracking-widest">
                                Active Workbench
                            </h2>
                            <button
                                onClick={() => navigate('/projects')}
                                className="text-[14px] leading-[20px] text-body hover:text-ink transition-colors flex items-center gap-1"
                            >
                                View all <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="bg-canvas rounded-[8px] border border-hairline shadow-[0px_1px_1px_#00000005] overflow-hidden">
                            {loading ? (
                                <div className="flex justify-center items-center p-12">
                                    <Loader2 className="w-6 h-6 animate-spin text-ink" />
                                </div>
                            ) : recentProjects.length === 0 ? (
                                <div className="p-12 text-center border-t border-hairline">
                                    <p className="text-[14px] text-mute">No active projects. Start a new workflow to see it here.</p>
                                </div>
                            ) : (
                                <div className="overflow-x-auto">
                                    <table className="w-full text-left border-collapse min-w-[500px]">
                                        <thead>
                                            <tr className="bg-canvas-soft border-b border-hairline">
                                                <th className="font-mono text-[12px] font-normal text-mute uppercase tracking-wider py-3 px-4">Project</th>
                                                <th className="font-mono text-[12px] font-normal text-mute uppercase tracking-wider py-3 px-4 hidden sm:table-cell">Mode</th>
                                                <th className="font-mono text-[12px] font-normal text-mute uppercase tracking-wider py-3 px-4">Status</th>
                                                <th className="font-mono text-[12px] font-normal text-mute uppercase tracking-wider py-3 px-4 hidden md:table-cell text-right">Last Updated</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-[#ebebeb]">
                                            {recentProjects.map(project => (
                                                <tr
                                                    key={project.id}
                                                    onClick={() => navigate(`/projects/${project.id}`)}
                                                    className="hover:bg-canvas-soft transition-colors cursor-pointer group"
                                                >
                                                    <td className="py-4 px-4 text-[14px] leading-[20px] font-medium text-ink">
                                                        {project.name}
                                                    </td>
                                                    <td className="py-4 px-4 text-[14px] leading-[20px] text-body hidden sm:table-cell">
                                                        <span className="capitalize">{project.mode.replace('_', ' ')}</span>
                                                    </td>
                                                    <td className="py-4 px-4">
                                                        <div className="flex items-center gap-2">
                                                            <div className={`w-2 h-2 rounded-full ${getStatusColor(project.status)}`} />
                                                            <span className="text-[14px] leading-[20px] text-body capitalize">{project.status}</span>
                                                        </div>
                                                    </td>
                                                    <td className="py-4 px-4 text-[14px] leading-[20px] text-mute hidden md:table-cell text-right">
                                                        {new Date(project.createdAt).toLocaleDateString()}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </section>

                    {/* ── Section 5: Pipeline Activity ── */}
                    <section>
                        <h2 className="font-mono text-[12px] leading-[16px] uppercase text-mute tracking-widest mb-6">
                            Activity
                        </h2>
                        <ActivityTimeline items={timelineLogs} />
                    </section>
                </div>

                {/* ── Section 4: Quick Start ── */}
                <section>
                    <h2 className="font-mono text-[12px] leading-[16px] uppercase text-mute tracking-widest mb-6">
                        Quick Start
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[
                            {
                                title: 'Generate Dataset',
                                desc: 'Create training-ready datasets from documents.',
                                icon: Database,
                                mode: 'dataset_only'
                            },
                            {
                                title: 'Fine-Tune Model',
                                desc: 'Build an AI model tailored to your domain.',
                                icon: Brain,
                                mode: 'finetune_only'
                            },
                            {
                                title: 'Deploy Agents',
                                desc: 'Create specialized AI agents in minutes.',
                                icon: Server,
                                mode: 'deploy_agents'
                            },
                            {
                                title: 'Generate Automations',
                                desc: 'Transform processes into automated workflows.',
                                icon: Zap,
                                mode: 'generate_automations'
                            },
                            {
                                title: 'Full Pipeline',
                                desc: 'Build your complete AI ecosystem.',
                                icon: Layers,
                                mode: 'full'
                            }
                        ].map((card, i) => (
                            <SpotlightCard
                                key={i}
                                onClick={() => navigate(`/projects/new?mode=${card.mode}`)}
                                className="group bg-canvas rounded-[8px] p-6 border border-hairline shadow-[0px_1px_1px_#00000005] hover:shadow-[0px_1px_1px_#00000005,0px_2px_2px_#0000000a,0px_8px_8px_-8px_#0000000a] hover:border-hairline-strong transition-all cursor-pointer flex flex-col"
                            >
                                <div className="w-10 h-10 rounded-full border border-hairline bg-canvas-soft flex items-center justify-center mb-6 group-hover:bg-ink transition-colors">
                                    <card.icon className="w-4 h-4 text-ink group-hover:text-canvas transition-colors" />
                                </div>
                                <h3 className="text-[16px] leading-[24px] font-medium text-ink mb-1">
                                    {card.title}
                                </h3>
                                <p className="text-[14px] leading-[20px] text-body">
                                    {card.desc}
                                </p>
                            </SpotlightCard>
                        ))}
                    </div>
                </section>

            </div>
        </div>
    );
}

// ─── Animated Pipeline Diagram ───────────────────────────────────────────────
const PIPELINE_STEPS = [
    { icon: FileText, label: 'Docs' },
    { icon: Database, label: 'Data' },
    { icon: Target, label: 'Eval' },
    { icon: Brain, label: 'Train' },
    { icon: Rocket, label: 'Deploy' },
];

const TRAVEL_MS   = 900;   // signal traverses a connector
const HOLD_MS     = 2500;  // node stays lit
const PAUSE_MS    = 1200;  // pause after last node before restart

function PipelineDiagram() {
    // activeNode  : which node is glowing  (-1 = none, traveling)
    // travelIdx   : which connector is showing the pulse (0-3)
    const [activeNode,  setActiveNode]  = useState(0);
    const [travelIdx,   setTravelIdx]   = useState(null);
    const timerRef = useRef([]);

    const clearTimers = () => { timerRef.current.forEach(clearTimeout); timerRef.current = []; };

    const schedule = useCallback(() => {
        clearTimers();
        let cursor = 0;

        const step = (nodeIdx) => {
            // Arrive at node
            const t1 = setTimeout(() => {
                setTravelIdx(null);
                setActiveNode(nodeIdx);
            }, 0);

            // After hold, either travel to next or restart
            const t2 = setTimeout(() => {
                if (nodeIdx < PIPELINE_STEPS.length - 1) {
                    setActiveNode(null);          // dim current node
                    setTravelIdx(nodeIdx);        // show connector pulse
                    const t3 = setTimeout(() => step(nodeIdx + 1), TRAVEL_MS);
                    timerRef.current.push(t3);
                } else {
                    // Last node — pause, then restart
                    setActiveNode(null);
                    const t4 = setTimeout(() => {
                        setTravelIdx(null);
                        step(0);
                    }, PAUSE_MS);
                    timerRef.current.push(t4);
                }
            }, HOLD_MS);

            timerRef.current.push(t1, t2);
        };

        step(0);
    }, []);

    useEffect(() => { schedule(); return () => clearTimers(); }, [schedule]);

    return (
        <div className="hidden lg:flex items-center gap-4 py-8 px-10 bg-canvas rounded-[12px] border border-hairline shadow-[0px_1px_1px_#00000005,0px_8px_16px_-4px_#0000000a] relative overflow-hidden">
            {PIPELINE_STEPS.map((step, idx, arr) => (
                <React.Fragment key={idx}>
                    {/* Node */}
                    <div className="flex flex-col items-center gap-3">
                        <div className="relative w-12 h-12">
                            {/* Decorative Glow Layer */}
                            {activeNode === idx && (
                                <>
                                    {/* Outset ring */}
                                    <div className="absolute inset-[-3px] rounded-full border border-[rgba(0,223,216,0.12)] pointer-events-none z-0" />
                                    {/* Blurred back-glow */}
                                    <div
                                        className="absolute inset-[-8px] rounded-full pointer-events-none z-0"
                                        style={{
                                            background: 'rgba(0,223,216,0.15)',
                                            filter: 'blur(8px)',
                                        }}
                                    />
                                </>
                            )}

                            {/* Crisp Circle Node itself */}
                            <div
                                className="absolute inset-0 rounded-full border flex items-center justify-center transition-all duration-500 bg-canvas-soft z-10"
                                style={{
                                    borderColor:    activeNode === idx ? '#00DFD8' : 'var(--hairline)',
                                    backgroundColor: activeNode === idx ? 'rgba(0,223,216,0.08)' : 'var(--canvas-soft)',
                                }}
                            >
                                <step.icon
                                    className="w-5 h-5 transition-colors duration-500"
                                    style={{ color: activeNode === idx ? '#00DFD8' : 'var(--ink)' }}
                                    strokeWidth={1.5}
                                />
                            </div>
                        </div>
                        <span
                            className="font-mono text-[10px] uppercase tracking-wider transition-colors duration-500"
                            style={{ color: activeNode === idx ? '#00DFD8' : 'var(--mute)' }}
                        >
                            {step.label}
                        </span>
                    </div>

                    {/* Connector with traveling pulse */}
                    {idx < arr.length - 1 && (
                        <div className="relative w-8 -mt-6" style={{ height: '1px' }}>
                            {/* static dashed baseline */}
                            <div className="absolute inset-0 border-t border-dashed border-hairline-strong" />
                            {/* traveling pulse dot */}
                            {travelIdx === idx && (
                                <div
                                    className="absolute top-[-3px] h-[7px] rounded-full"
                                    style={{
                                        width: '12px',
                                        background: 'linear-gradient(90deg, transparent 0%, #00DFD8 50%, transparent 100%)',
                                        animation: `pipeline-travel ${TRAVEL_MS}ms cubic-bezier(0.25,0,0.25,1) forwards`,
                                    }}
                                />
                            )}
                        </div>
                    )}
                </React.Fragment>
            ))}

            <style>{`
                @keyframes pipeline-travel {
                    from { left: 0px;  opacity: 0; }
                    15%  { opacity: 1; }
                    85%  { opacity: 1; }
                    to   { left: calc(100% - 12px); opacity: 0; }
                }
            `}</style>
        </div>
    );
}

// ─── Animated Activity Timeline ───────────────────────────────────────────────
function ActivityTimeline({ items }) {
    const [activeItem,  setActiveItem]  = useState(0);
    const [travelLine,  setTravelLine]  = useState(null); // connector index showing pulse
    const timerRef = useRef([]);

    const clearTimers = () => { timerRef.current.forEach(clearTimeout); timerRef.current = []; };

    const schedule = useCallback(() => {
        clearTimers();

        const step = (itemIdx) => {
            const t1 = setTimeout(() => {
                setTravelLine(null);
                setActiveItem(itemIdx);
            }, 0);

            const t2 = setTimeout(() => {
                if (itemIdx < items.length - 1) {
                    setActiveItem(null);
                    setTravelLine(itemIdx);
                    const t3 = setTimeout(() => step(itemIdx + 1), TRAVEL_MS);
                    timerRef.current.push(t3);
                } else {
                    setActiveItem(null);
                    const t4 = setTimeout(() => { setTravelLine(null); step(0); }, PAUSE_MS);
                    timerRef.current.push(t4);
                }
            }, HOLD_MS);

            timerRef.current.push(t1, t2);
        };

        step(0);
    }, [items]);

    useEffect(() => { schedule(); return () => clearTimers(); }, [schedule]);

    return (
        <div className="bg-canvas rounded-[8px] border border-hairline shadow-[0px_1px_1px_#00000005] p-6 h-[calc(100%-40px)]">
            <div className="space-y-6">
                {items.map((item, i) => (
                    <div key={item.id} className="relative pl-6">
                        {/* Vertical connector */}
                        {i !== items.length - 1 && (
                            <div className="absolute left-[7px] top-6 bottom-[-24px] w-[1px] bg-hairline overflow-hidden">
                                {travelLine === i && (
                                    <div
                                        className="absolute left-0 w-full rounded-full"
                                        style={{
                                            height: '16px',
                                            background: 'linear-gradient(180deg, transparent 0%, #7DD3FC 50%, transparent 100%)',
                                            animation: `timeline-travel ${TRAVEL_MS}ms cubic-bezier(0.25,0,0.25,1) forwards`,
                                        }}
                                    />
                                )}
                            </div>
                        )}

                        {/* Node indicator */}
                        <div className="absolute left-0 top-1.5 w-4 h-4">
                            {/* Decorative Glow Layer */}
                            {activeItem === i && (
                                <>
                                    {/* Outset ring */}
                                    <div className="absolute inset-[-2px] rounded-full border border-[rgba(125,211,252,0.15)] pointer-events-none z-0" />
                                    {/* Blurred back-glow */}
                                    <div
                                        className="absolute inset-[-4px] rounded-full pointer-events-none z-0"
                                        style={{
                                            background: 'rgba(125,211,252,0.15)',
                                            filter: 'blur(4px)',
                                        }}
                                    />
                                </>
                            )}

                            {/* Crisp Circle Node itself */}
                            <div
                                className="absolute inset-0 rounded-full border flex items-center justify-center transition-all duration-500 bg-canvas z-10"
                                style={{
                                    borderColor: activeItem === i ? '#7DD3FC' : 'var(--hairline)',
                                }}
                            >
                                <div
                                    className="w-1.5 h-1.5 rounded-full transition-colors duration-500"
                                    style={{ background: activeItem === i ? '#7DD3FC' : 'var(--ink)' }}
                                />
                            </div>
                        </div>

                        {/* Text */}
                        <div className="flex flex-col gap-0.5">
                            <span
                                className="text-[14px] leading-[20px] font-medium transition-colors duration-500"
                                style={{ color: activeItem === i ? '#7DD3FC' : 'var(--ink)' }}
                            >
                                {item.event}
                            </span>
                            <span className="font-mono text-[12px] leading-[16px] text-mute">
                                {item.time}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            <style>{`
                @keyframes timeline-travel {
                    from { top: -16px; opacity: 0; }
                    15%  { opacity: 1; }
                    85%  { opacity: 1; }
                    to   { top: 100%;  opacity: 0; }
                }
            `}</style>
        </div>
    );
}
