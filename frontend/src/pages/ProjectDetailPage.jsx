import React, { useState, useEffect, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
    ArrowLeft,
    Clock,
    FileText,
    Brain,
    Upload,
    ScanText,
    Layers,
    Database,
    ShieldCheck,
    Rocket,
    RotateCcw,
    Trash2,
    CheckCircle2,
    AlertCircle,
    Loader2,
    Pause,
    Download,
    Circle,
    ExternalLink,
    BarChart3,
    Timer,
    Settings2,
    Terminal,
    Bot,
    Zap,
    Server,
    ArrowDown,
    Copy,
    Check,
    ChevronDown,
    ChevronUp,
    Eye,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogClose,
} from '@/components/ui/dialog'
import PipelineTracker from '@/components/PipelineTracker'
import { getStepsForMode } from '../config/pipeline'
import { apiClient } from '@/api/client'

const stateToStepMap = {
    'AnalyzeIntentAndStrategize': 'upload',
    'CheckDatasetGenNeeded': 'upload',
    'DocumentProcessing': 'ocr',
    'OCR': 'ocr',
    'Chunking': 'chunking',
    'GenerateDatasetMap': 'chunking',
    'PrepareMapInput': 'dataset_gen',
    'DatasetGeneration': 'dataset_gen',
    'GenerateSamplesForChunk': 'dataset_gen',
    'Collector': 'dataset_gen',
    'QualityControl': 'quality_control',
    'EvaluateDatasetQuality': 'quality_control',
    'StartFineTuningJob': 'fine_tuning',
    'CheckPollLimit': 'fine_tuning',
    'CheckTrainingStatus': 'fine_tuning',
    'IsTrainingComplete': 'fine_tuning',
    'EvaluateModelPerformance': 'fine_tuning',
    'AgenticDecisionNode': 'fine_tuning',
    'ExecuteAgentDecision': 'fine_tuning',
    'FineTuning': 'fine_tuning',
    'DeployEndpoint': 'deployment',
    'CheckDeployNeeded': 'deployment',
    'Deployment': 'deployment',
    'KnowledgeExtraction': 'knowledge_extraction',
    'AgentDiscovery': 'agent_discovery',
    'AutomationDiscovery': 'automation_discovery',
    'AgentDeployment': 'agent_deployment'
}

const LiveTimer = ({ startTime }) => {
    const [now, setNow] = useState(Date.now())
    useEffect(() => {
        const interval = setInterval(() => setNow(Date.now()), 1000)
        return () => clearInterval(interval)
    }, [])
    
    const ms = Math.max(0, now - startTime)
    const seconds = Math.floor(ms / 1000)
    const m = Math.floor(seconds / 60)
    const s = seconds % 60
    if (m > 0) return `${m}m ${s}s`
    return `${s}s`
}

const stepIconMap = { Upload, ScanText, Layers, Database, ShieldCheck, Brain, Rocket, Bot, Zap, Server }

const statusConfig = {
    running: { label: 'Running', className: 'bg-blue-500/15 text-blue-400 border-blue-500/30', icon: Loader2 },
    completed: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', icon: CheckCircle2 },
    complete: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', icon: CheckCircle2 },
    failed: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30', icon: AlertCircle },
    error: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30', icon: AlertCircle },
    pending: { label: 'Pending', className: 'bg-amber-500/15 text-amber-400 border-amber-500/30', icon: Pause },
}

const stepDetails = {
    complete: { duration: '2m 34s', output: 'Completed successfully' },
    running: { duration: 'In progress...', output: 'Processing data...' },
    error: { duration: '1m 12s', output: 'Failed: Check logs for details.' },
    pending: { duration: '—', output: 'Waiting for previous steps to complete' },
}

// Build human-readable summary lines for each step's result data
function buildStepSummary(stepName, data) {
    if (!data || typeof data !== 'object') return []
    const lines = []

    switch (stepName) {
        case 'upload':
            if (data.raw_file_keys?.length) lines.push(`📁 ${data.raw_file_keys.length} file(s) uploaded`)
            break
        case 'ocr':
            if (data.files_processed != null) lines.push(`📄 ${data.files_processed} file(s) processed`)
            if (data.total_characters != null) lines.push(`✏️ ${data.total_characters.toLocaleString()} characters extracted`)
            if (data.files_failed > 0) lines.push(`⚠️ ${data.files_failed} file(s) failed`)
            if (data.errors?.length) {
                data.errors.forEach(e => lines.push(`❌ ${e.file}: ${e.error}`))
            }
            break
        case 'chunking':
            if (data.chunk_count != null) lines.push(`🧩 ${data.chunk_count} chunks created`)
            if (data.total_words != null) lines.push(`📝 ${data.total_words.toLocaleString()} total words`)
            break
        case 'generation':
            if (data.example_count != null) lines.push(`🤖 ${data.example_count} training examples generated`)
            if (data.chunks_processed != null) lines.push(`✅ ${data.chunks_processed} chunks processed`)
            if (data.chunks_failed > 0) lines.push(`⚠️ ${data.chunks_failed} chunks failed`)
            break
        case 'quality_control':
            if (data.total_input != null) lines.push(`📊 ${data.total_input} examples evaluated`)
            if (data.kept != null) lines.push(`✅ ${data.kept} kept (above ${(data.threshold || 0.7) * 100}% threshold)`)
            if (data.discarded > 0) lines.push(`🗑️ ${data.discarded} discarded`)
            if (data.duplicates_removed > 0) lines.push(`♻️ ${data.duplicates_removed} duplicates removed`)
            break
        case 'fine_tuning':
            if (data.job_name) lines.push(`🔧 Job: ${data.job_name}`)
            if (data.duration_min) lines.push(`⏱️ ${data.duration_min} minutes`)
            if (data.final_loss != null) lines.push(`📉 Final loss: ${data.final_loss}`)
            break
        case 'deployment':
            if (data.endpoint_url) lines.push(`🚀 Endpoint: ${data.endpoint_url}`)
            break
        case 'knowledge_extraction':
            if (data.domains_found != null) lines.push(`🧠 ${data.domains_found} knowledge domains identified`)
            if (data.workflows_found != null) lines.push(`📋 ${data.workflows_found} internal workflows mapped`)
            break
        case 'agent_discovery':
            if (data.agents_discovered != null) lines.push(`🤖 ${data.agents_discovered} specialized agents discovered`)
            if (data.roles?.length) data.roles.forEach(r => lines.push(`• ${r}`))
            break
        case 'automation_discovery':
            if (data.automations_found != null) lines.push(`⚡ ${data.automations_found} n8n automations discovered`)
            break
        case 'agent_deployment':
            if (data.deployed_agents != null) lines.push(`🚀 ${data.deployed_agents} agents deployed to FastAPI`)
            if (data.endpoint_url) lines.push(`🌐 Base URL: ${data.endpoint_url}`)
            break
        default:
            // Generic: show any keys with values
            Object.entries(data).forEach(([k, v]) => {
                if (v != null && typeof v !== 'object') lines.push(`${k}: ${v}`)
            })
    }
    return lines
}

// ── Log Formatter ─────────────────────────────────────────────────────────────
function formatLogEntry(log) {
    const { type, label, summary, details } = log
    
    // Ignore noisy events
    const noisyTypes = ['PassStateEntered', 'PassStateExited', 'WaitStateEntered', 'WaitStateExited']
    const noisyLabels = ['CheckPollLimit', 'CheckTrainingStatus', 'IsTrainingComplete', 'EvaluateModelPerformance', 'AgenticDecisionNode', 'ExecuteAgentDecision']
    
    if (noisyTypes.includes(type)) return null
    if (noisyLabels.some(s => label.includes(s))) return null

    let message = label
    let status = 'info'
    let icon = '•'

    if (type.includes('Failed') || type.includes('Aborted') || type.includes('TimedOut')) {
        status = 'error'
        icon = '✕'
        message = summary || `Failed during ${label}`
    } else if (type.includes('Succeeded') || type.includes('Succeed')) {
        status = 'success'
        icon = '✓'
        if (type === 'ExecutionSucceeded') message = 'Pipeline completed successfully'
    } else if (type.includes('Started') || type.includes('Entered')) {
        status = 'running'
        icon = '→'
        if (type === 'ExecutionStarted') message = 'Pipeline execution initiated'
    }

    // Clean up Step Functions specific labels
    const cleanLabel = label
        .replace('Entered: ', 'Starting ')
        .replace('Exited: ', 'Finished ')
        .replace(/_/g, ' ')

    return {
        id: log.id,
        timestamp: log.timestamp,
        message: cleanLabel,
        detail: summary,
        status,
        icon,
        raw: details
    }
}

// ── n8n Automation Generator & UI ─────────────────────────────────────────────
const generateN8nWorkflow = (auto) => {
    const nodes = [];
    const connections = {};
    
    // Trigger
    const triggerId = "trigger-1";
    const triggerName = `Trigger: ${auto.automation_blueprint?.trigger || 'Event'}`;
    nodes.push({
        parameters: {},
        id: triggerId,
        name: triggerName,
        type: "n8n-nodes-base.manualTrigger",
        typeVersion: 1,
        position: [200, 300]
    });
    
    let previousName = triggerName;
    connections[previousName] = { main: [[]] };

    // Actions
    const actions = auto.automation_blueprint?.actions || [];
    actions.forEach((action, index) => {
        const actionId = `action-${index + 1}`;
        const actionName = action;
        
        nodes.push({
            parameters: {},
            id: actionId,
            name: actionName,
            type: "n8n-nodes-base.noOp",
            typeVersion: 1,
            position: [400 + (index * 200), 300]
        });
        
        connections[previousName].main[0].push({
            node: actionName,
            type: "main",
            index: 0
        });
        
        previousName = actionName;
        connections[previousName] = { main: [[]] };
    });

    return {
        name: auto.name,
        nodes: nodes,
        connections: connections,
        active: false,
        settings: {},
        versionId: "1.0"
    };
};

const AutomationCard = ({ auto }) => {
    const [showPreview, setShowPreview] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [copied, setCopied] = useState(false);

    const cachedWorkflow = useMemo(() => generateN8nWorkflow(auto), [auto]);
    const cachedJson = useMemo(() => JSON.stringify(cachedWorkflow, null, 2), [cachedWorkflow]);

    const handleCopyToClipboard = async () => {
        try {
            await navigator.clipboard.writeText(cachedJson);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error("Failed to copy JSON to clipboard", err);
        }
    };

    const handleImportClick = async () => {
        // Future REST API integration point:
        // if (userSettings.n8nApiKey) { await deployViaRest(cachedWorkflow); return window.open(n8nUrl); }
        await handleCopyToClipboard();
        setShowModal(true);
    };

    const handleDownload = () => {
        const blob = new Blob([cachedJson], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${auto.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="flex flex-col gap-2 text-xs border border-border/50 rounded-md bg-muted/20">
            <div className="p-3">
                <span className="font-semibold text-foreground block mb-1">{auto.name}</span>
                <p className="text-muted-foreground/80 leading-relaxed mb-3 line-clamp-3" title={auto.description}>{auto.description}</p>
                
                {/* Metrics */}
                <div className="flex flex-wrap gap-2 mb-3">
                    <Badge variant="outline" className="text-[10px] bg-background">Impact: {auto.business_impact}</Badge>
                    <Badge variant="outline" className="text-[10px] bg-background">Score: {auto.automation_score}/100</Badge>
                </div>

                <div className="flex items-center gap-2">
                    <Button
                        size="sm"
                        className="flex-1 gap-1.5 h-8 text-[11px]"
                        onClick={handleImportClick}
                    >
                        <ExternalLink className="w-3.5 h-3.5" />
                        Import into n8n
                    </Button>
                    <Button
                        size="sm"
                        variant="ghost"
                        className="h-8 w-8 p-0 shrink-0"
                        onClick={() => setShowPreview(!showPreview)}
                        title="View Workflow Preview"
                    >
                        {showPreview ? <ChevronUp className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </Button>
                </div>
            </div>

            {/* Workflow Preview UI */}
            {showPreview && (
                <div className="px-3 pb-3 pt-1 border-t border-border/50 bg-background/50 animate-in fade-in slide-in-from-top-1">
                    <p className="text-[10px] font-mono uppercase text-muted-foreground mb-2">Workflow Blueprint</p>
                    <div className="space-y-1.5 pl-2">
                        <div className="flex items-start gap-2">
                            <div className="mt-0.5 w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0"></div>
                            <span className="font-medium text-[11px] leading-tight">{auto.automation_blueprint?.trigger}</span>
                        </div>
                        {auto.automation_blueprint?.actions?.map((action, idx) => (
                            <React.Fragment key={idx}>
                                <div className="pl-0.5 my-0.5 text-muted-foreground/40">
                                    <ArrowDown className="w-3 h-3" />
                                </div>
                                <div className="flex items-start gap-2">
                                    <div className="mt-0.5 w-1.5 h-1.5 rounded-full bg-zinc-400 shrink-0 border border-border"></div>
                                    <span className="text-muted-foreground text-[11px] leading-tight">{action}</span>
                                </div>
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            )}

            {/* Helper Modal */}
            <Dialog open={showModal} onOpenChange={setShowModal}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <CheckCircle2 className="w-5 h-5 text-green-500" />
                            Workflow Ready
                        </DialogTitle>
                        <DialogDescription className="pt-2">
                            Your generated workflow has been copied to the clipboard.
                        </DialogDescription>
                    </DialogHeader>
                    
                    <div className="py-4">
                        <h4 className="text-sm font-semibold mb-2 text-foreground">Next Steps:</h4>
                        <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                            <li>Click "Open n8n" below.</li>
                            <li>Click anywhere on the blank canvas.</li>
                            <li>Press <kbd className="px-1.5 py-0.5 bg-muted rounded border font-mono">Ctrl+V</kbd> (Cmd+V on Mac).</li>
                            <li>The workflow will instantly appear.</li>
                        </ol>
                    </div>

                    <DialogFooter className="flex-col sm:flex-row gap-2 mt-2">
                        <Button variant="outline" onClick={handleDownload} className="w-full sm:w-auto gap-2 text-xs">
                            <Download className="w-3.5 h-3.5" />
                            Download JSON
                        </Button>
                        <Button variant="secondary" onClick={handleCopyToClipboard} className="w-full sm:w-auto gap-2 text-xs transition-all">
                            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
                            {copied ? "Copied!" : "Copy Again"}
                        </Button>
                        <Button onClick={() => window.open('https://app.n8n.cloud/', '_blank')} className="w-full sm:w-auto gap-2 text-xs bg-[#FF6E57] hover:bg-[#e05b45] text-white">
                            <ExternalLink className="w-3.5 h-3.5" />
                            Open n8n
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default function ProjectDetailPage() {
    const { id } = useParams()
    const navigate = useNavigate()
    const [project, setProject] = useState(null)
    const [pipelineStatus, setPipelineStatus] = useState('NOT_STARTED')
    const [results, setResults] = useState(null)
    const [activeStep, setActiveStep] = useState(null)
    const [loading, setLoading] = useState(true)
    const [deleting, setDeleting] = useState(false)
    const [showDeleteDialog, setShowDeleteDialog] = useState(false)
    const [logs, setLogs] = useState([])   // full list – used for duration math
    const [showLogs, setShowLogs] = useState(false)
    const [fetchingLogs, setFetchingLogs] = useState(false)
    const [logError, setLogError] = useState(false)

    // Memoize the display-ready log list so LiveTimer ticks don't reprocess it
    const displayLogs = useMemo(() => {
        return logs.slice(0, 200).map(log => formatLogEntry(log)).filter(Boolean)
    }, [logs])

    // ── Fetch project metadata ────────────────────────────────────────────────
    useEffect(() => {
        const fetchProject = async () => {
            try {
                const data = await apiClient.get(`projects/${id}`)
                setProject({
                    id: data.id,
                    name: data.name,
                    description: data.description,
                    status: data.status,
                    mode: data.mode,
                    createdAt: data.created_at,
                    model: data.base_model || 'Unknown',
                    intent: data.intent,
                })
            } catch (err) {
                console.error(err)
            } finally {
                setLoading(false)
            }
        }
        fetchProject()
    }, [id])

    // ── Poll status & logs ────────────────────────────────────────────────────
    useEffect(() => {
        if (!project) return

        const checkStatus = async () => {
            try {
                const statusData = await apiClient.get(`projects/${id}/status`)
                const rawStatus = statusData.status || statusData.pipeline_status || 'NOT_STARTED'
                const normalizedStatus = rawStatus === 'COMPLETE' ? 'SUCCEEDED' : rawStatus
                setPipelineStatus(normalizedStatus)
                setProject(prev => ({ ...prev, status: statusData.status || statusData.project_status || prev.status }))
            } catch (err) {
                console.error("Status error", err)
            }
        }

        checkStatus()

        let interval
        if (pipelineStatus === 'RUNNING') {
            interval = setInterval(checkStatus, 5000)
        }

        return () => clearInterval(interval)
    }, [id, project?.id, pipelineStatus])

    // ── Fetch logs ───────────────────────────────────────────────────────────
    const fetchLogs = async () => {
        if (fetchingLogs) return  // don't stack requests
        try {
            setFetchingLogs(true)
            setLogError(false)
            const data = await apiClient.get(`projects/${id}/logs`)
            setLogs(data.logs || [])
        } catch (err) {
            // Silently track error state; don't spam console every 5s
            setLogError(true)
        } finally {
            setFetchingLogs(false)
        }
    }

    useEffect(() => {
        if (!id) return
        if (showLogs || pipelineStatus !== 'NOT_STARTED') {
            fetchLogs()
            // If running, poll logs frequently to keep progress accurate
            if (pipelineStatus === 'RUNNING') {
                const interval = setInterval(fetchLogs, 5000)
                return () => clearInterval(interval)
            }
        }
    }, [id, showLogs, pipelineStatus])

    // ── Fetch results when pipeline finishes ──────────────────────────────────
    useEffect(() => {
        if (pipelineStatus !== 'SUCCEEDED' && pipelineStatus !== 'FAILED') return

        const fetchResults = async () => {
            try {
                const data = await apiClient.get(`projects/${id}/results`)
                setResults(data)
            } catch (err) {
                console.error("Results fetch error", err)
            }
        }
        fetchResults()
    }, [id, pipelineStatus])

    // ── Delete project ────────────────────────────────────────────────────────
    const handleDelete = async () => {
        try {
            setDeleting(true)
            await apiClient.delete(`projects/${id}`)
            setShowDeleteDialog(false)
            navigate('/projects')
        } catch (err) {
            console.error('Delete error:', err)
        } finally {
            setDeleting(false)
        }
    }

    // ── Retry project ─────────────────────────────────────────────────────────
    const handleRetry = async () => {
        try {
            await apiClient.post(`projects/${id}/start`, {})
            setPipelineStatus('RUNNING')
            setProject(prev => ({ ...prev, status: 'running' }))
            setResults(null)
            setLogs([])
        } catch (err) {
            console.error('Failed to retry pipeline:', err)
            // Could add a toast notification here if available
        }
    }

    // ── Mode-based steps ────────────────────────────────────────────────────────
    const modeSteps = getStepsForMode(project?.mode)
    const stepCount = modeSteps.length

    // ── Derive pipeline progress array ────────────────────────────────────────
    const derivePipelineArr = () => {
        const arr = []
        const isRun = pipelineStatus === 'RUNNING'
        const isSuc = pipelineStatus === 'SUCCEEDED'
        const isErr = pipelineStatus === 'FAILED'

        // Map Step Functions state names to our local step IDs (moved to top of file)

        // Find the current active step index based on logs
        let activeIdx = -1
        if (isRun || isErr) {
            // Find the most recent "Entered" state in logs
            const enteringLogs = logs.filter(l => l.type === 'TaskStateEntered' || l.type === 'ChoiceStateEntered' || l.label.startsWith('Entered:'))
            
            // Search backwards (newest first) for the first mapped state
            for (const log of enteringLogs) {
                const stateName = log.label.replace('Entered: ', '')
                const stepId = stateToStepMap[stateName]
                if (stepId) {
                    activeIdx = modeSteps.findIndex(s => s.id === stepId)
                    if (activeIdx !== -1) {
                        break // Found the most recent active mapped step!
                    }
                }
            }
        }

        for (let i = 0; i < stepCount; i++) {
            if (isSuc) {
                arr.push({ step: i + 1, status: 'complete', progress: 100 })
            } else if (isErr) {
                // If we know where it failed, mark that step as error
                if (activeIdx !== -1) {
                    if (i < activeIdx) arr.push({ step: i + 1, status: 'complete', progress: 100 })
                    else if (i === activeIdx) arr.push({ step: i + 1, status: 'error', progress: 50 })
                    else arr.push({ step: i + 1, status: 'pending', progress: 0 })
                } else {
                    // Fallback: mark the last step as error
                    if (i === stepCount - 1) arr.push({ step: i + 1, status: 'error', progress: 50 })
                    else arr.push({ step: i + 1, status: 'complete', progress: 100 })
                }
            } else if (isRun) {
            // Calculate a dynamic progress based on recent log activity
            let calculatedProgress = 75
            if (activeIdx !== -1) {
                const activeStepDef = modeSteps[activeIdx]
                let activeLogsCount = 0
                for (const log of logs) {
                    const stateName = log.label.replace('Entered: ', '').replace('Exited: ', '')
                    const stepId = stateToStepMap[stateName]
                    if (stepId === activeStepDef.id) activeLogsCount++
                    else if (stepId && stepId !== activeStepDef.id) break // Reached previous step
                }
                calculatedProgress = Math.min(15 + (activeLogsCount * 3), 95) // Max 95%
            }

            // Use detected active step, or fallback to first step
            const currentIdx = activeIdx !== -1 ? activeIdx : 0
            if (i < currentIdx) arr.push({ step: i + 1, status: 'complete', progress: 100 })
            else if (i === currentIdx) arr.push({ step: i + 1, status: 'running', progress: calculatedProgress })
            else arr.push({ step: i + 1, status: 'pending', progress: 0 })
            } else {
                arr.push({ step: i + 1, status: 'pending', progress: 0 })
            }
        }
        return arr
    }

    // ── Loading / NotFound ────────────────────────────────────────────────────
    if (loading) {
        return <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>
    }

    if (!project) {
        return (
            <div className="flex flex-col items-center justify-center h-full py-20">
                <p className="text-lg font-medium">Project not found</p>
                <p className="text-sm text-muted-foreground mt-1 mb-4">The project you're looking for doesn't exist</p>
                <Button variant="outline" onClick={() => navigate('/projects')}>← Back to Projects</Button>
            </div>
        )
    }

    const pipelineArr = derivePipelineArr()
    const uiStatusKey = pipelineStatus === 'SUCCEEDED' ? 'completed' : pipelineStatus === 'FAILED' ? 'failed' : pipelineStatus === 'RUNNING' ? 'running' : 'pending'
    const status = statusConfig[uiStatusKey] || statusConfig.pending
    const StatusIcon = status?.icon || Clock
    const completedSteps = pipelineArr.filter(s => s.status === 'complete').length
    const overallProgress = Math.round((completedSteps / stepCount) * 100)
    const displayStep = activeStep !== null ? activeStep : Math.min(completedSteps, stepCount - 1)
    const currentPipelineStep = pipelineArr[displayStep] || { status: 'pending', progress: 0 }
    const currentStepDef = modeSteps[displayStep]
    const StepIcon = stepIconMap[currentStepDef?.icon] || Circle
    const detail = stepDetails[currentPipelineStep.status] || stepDetails.pending

    // Use specific error message if available
    const errorDisplay = results?.error ? (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 mt-2">
            <p className="text-red-400 text-xs font-semibold uppercase tracking-wider mb-1">Pipeline Error</p>
            <p className="text-sm font-medium text-red-100">{results.error.error}</p>
            <p className="text-xs text-red-300/80 mt-1 font-mono">{results.error.cause}</p>
        </div>
    ) : detail.output

    const timeAgo = (dateStr) => {
        if (!dateStr) return 'Just now'
        const diff = Date.now() - new Date(dateStr).getTime()
        const days = Math.floor(diff / (1000 * 60 * 60 * 24))
        if (days === 0) return 'Today'
        if (days === 1) return 'Yesterday'
        return `${days} days ago`
    }

    const formatStaticDuration = (ms) => {
        if (ms < 0) ms = 0
        const seconds = Math.floor(ms / 1000)
        const m = Math.floor(seconds / 60)
        const s = seconds % 60
        if (m > 0) return `${m}m ${s}s`
        return `${s}s`
    }

    const renderDuration = (stepId, stepStatus) => {
        if (!logs || logs.length === 0 || stepStatus === 'pending') return 'Waiting...'
        
        const stepLogs = logs.filter(l => {
            if (!l.label) return false
            const stateName = l.label.replace('Entered: ', '').replace('Exited: ', '').replace('Starting ', '').replace('Finished ', '')
            return stateToStepMap[stateName] === stepId
        })
        
        if (stepLogs.length === 0) return stepStatus === 'running' ? 'Starting...' : 'Unknown'
        
        // Logs are newest first. Oldest is at the end.
        const startTime = new Date(stepLogs[stepLogs.length - 1].timestamp).getTime()
        
        if (stepStatus === 'complete' || stepStatus === 'error') {
            const endTime = new Date(stepLogs[0].timestamp).getTime()
            return formatStaticDuration(endTime - startTime)
        } else if (stepStatus === 'running') {
            return <LiveTimer startTime={startTime} />
        }
        return 'Unknown'
    }

    return (
        <div className="p-6 md:p-8 max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                <div className="flex items-start gap-3">
                    <Button variant="ghost" size="icon" className="h-8 w-8 mt-0.5" onClick={() => navigate('/projects')}>
                        <ArrowLeft className="w-4 h-4" />
                    </Button>
                    <div>
                        <div className="flex items-center gap-3">
                            <h1 className="text-2xl font-bold">{project.name}</h1>
                            <Badge variant="outline" className={`${status.className}`}>
                                <StatusIcon className={`w-3 h-3 mr-1 ${uiStatusKey === 'running' ? 'animate-spin' : ''}`} />
                                {status.label}
                            </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{project.description}</p>
                    </div>
                </div>

                <div className="flex items-center gap-2 sm:shrink-0">
                    {uiStatusKey === 'failed' && (
                        <Button variant="outline" size="sm" className="gap-1.5" onClick={handleRetry}>
                            <RotateCcw className="w-3.5 h-3.5" />
                            Retry
                        </Button>
                    )}
                    {results?.dataset_download_url && (
                        <Button
                            size="sm"
                            className="gap-1.5 bg-primary hover:bg-primary/90 text-primary-foreground"
                            onClick={() => window.open(results.dataset_download_url, '_blank')}
                        >
                            <Download className="w-3.5 h-3.5" />
                            Download Dataset
                        </Button>
                    )}
                    {uiStatusKey === 'completed' && (
                        <Button
                            variant="outline"
                            size="sm"
                            className="gap-1.5"
                            onClick={() => navigate(`/projects/${id}/dataset`)}
                        >
                            <Database className="w-3.5 h-3.5" />
                            Review Dataset
                        </Button>
                    )}
                    {uiStatusKey === 'completed' && (
                        <Button
                            size="sm"
                            className="gap-1.5 bg-primary hover:bg-primary/90 text-primary-foreground"
                            onClick={() => navigate(`/projects/${id}/compare`)}
                        >
                            <BarChart3 className="w-3.5 h-3.5" />
                            Compare Models
                        </Button>
                    )}
                    <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                        <DialogTrigger asChild>
                            <Button
                                variant="outline"
                                size="sm"
                                className="gap-1.5 text-destructive hover:text-destructive"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                                Delete
                            </Button>
                        </DialogTrigger>
                        <DialogContent className="sm:max-w-md">
                            <DialogHeader>
                                <DialogTitle>Delete Project</DialogTitle>
                                <DialogDescription>
                                    Are you sure you want to delete <span className="font-semibold text-foreground">{project.name}</span>? This will permanently remove the project, all uploaded files, and generated datasets. This action cannot be undone.
                                </DialogDescription>
                            </DialogHeader>
                            <DialogFooter className="gap-2 sm:gap-0">
                                <DialogClose asChild>
                                    <Button variant="outline" size="sm" disabled={deleting}>
                                        Cancel
                                    </Button>
                                </DialogClose>
                                <Button
                                    variant="destructive"
                                    size="sm"
                                    className="gap-1.5"
                                    onClick={handleDelete}
                                    disabled={deleting}
                                >
                                    {deleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                                    {deleting ? 'Deleting...' : 'Delete Project'}
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            {/* Project meta cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="rounded-lg border border-border bg-card px-4 py-3">
                    <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">Model</p>
                    <p className="text-sm font-semibold mt-1 flex items-center gap-1.5">
                        <Brain className="w-3.5 h-3.5 text-primary" />
                        {project.model}
                    </p>
                </div>
                <div className="rounded-lg border border-border bg-card px-4 py-3">
                    <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">Mode</p>
                    <p className="text-sm font-semibold mt-1 flex items-center gap-1.5">
                        <Settings2 className="w-3.5 h-3.5 text-primary" />
                        {project.mode?.replace(/_/g, ' ')}
                    </p>
                </div>
                <div className="rounded-lg border border-border bg-card px-4 py-3">
                    <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">Created</p>
                    <p className="text-sm font-semibold mt-1 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-primary" />
                        {timeAgo(project.createdAt)}
                    </p>
                </div>
                <div className="rounded-lg border border-border bg-card px-4 py-3">
                    <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">Progress</p>
                    <p className="text-sm font-semibold mt-1">{completedSteps}/{stepCount} steps ({overallProgress}%)</p>
                </div>
            </div>

            {/* Pipeline Tracker */}
            <Card className="border-border bg-card">
                <CardHeader className="pb-4">
                    <CardTitle className="text-base">Pipeline Progress</CardTitle>
                </CardHeader>
                <CardContent>
                    <PipelineTracker
                        steps={modeSteps}
                        pipeline={pipelineArr}
                        activeStep={displayStep}
                        onStepClick={setActiveStep}
                    />
                </CardContent>
            </Card>

            {/* Step Detail Panel */}
            <Card className="border-border bg-card">
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${currentPipelineStep.status === 'complete'
                                ? 'bg-emerald-500/15' : currentPipelineStep.status === 'running'
                                    ? 'bg-primary/15' : currentPipelineStep.status === 'error'
                                        ? 'bg-red-500/15' : 'bg-muted'
                                }`}>
                                <StepIcon className={`w-4 h-4 ${currentPipelineStep.status === 'complete'
                                    ? 'text-emerald-400' : currentPipelineStep.status === 'running'
                                        ? 'text-primary' : currentPipelineStep.status === 'error'
                                            ? 'text-red-400' : 'text-muted-foreground'
                                    }`} />
                            </div>
                            <div>
                                <CardTitle className="text-base">Step {displayStep + 1}: {currentStepDef.name}</CardTitle>
                                <p className="text-xs text-muted-foreground mt-0.5">{currentStepDef.description}</p>
                            </div>
                        </div>
                        <Badge variant="outline" className={`text-[11px] ${(statusConfig[currentPipelineStep.status] || statusConfig.pending).className}`}>
                            {(statusConfig[currentPipelineStep.status] || statusConfig.pending).label}
                        </Badge>
                    </div>
                </CardHeader>
                <CardContent className="space-y-4">
                    {currentPipelineStep.status === 'running' && (
                        <div className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-muted-foreground">Progress</span>
                                <span className="font-mono text-primary">{currentPipelineStep.progress}%</span>
                            </div>
                            <Progress value={currentPipelineStep.progress} className="h-2" />
                        </div>
                    )}
                    <Separator />
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                        <div>
                            <p className="text-muted-foreground text-xs mb-1">Duration</p>
                            <div className="font-medium">{renderDuration(currentStepDef?.id, currentPipelineStep.status)}</div>
                        </div>
                        <div>
                            <p className="text-muted-foreground text-xs mb-1">Status</p>
                            <div className={`font-medium ${currentPipelineStep.status === 'error' ? 'text-red-400' : ''}`}>
                                {currentPipelineStep.status === 'error' ? errorDisplay : detail.output}
                            </div>
                        </div>
                    </div>
                    {currentPipelineStep.status === 'error' && (
                        <>
                            <Separator />
                            <div className="flex items-center gap-2">
                                <Button variant="outline" size="sm" className="gap-1.5" onClick={handleRetry}>
                                    <RotateCcw className="w-3.5 h-3.5" />
                                    Retry This Step
                                </Button>
                            </div>
                        </>
                    )}
                </CardContent>
            </Card>

            {/* ── Results Panel ─────────────────────────────────────────────── */}
            {results && pipelineStatus === 'SUCCEEDED' && (
                <Card className="border-emerald-500/20 bg-card">
                    <CardHeader className="pb-3 border-b border-border">
                        <CardTitle className="text-base flex items-center gap-2">
                            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                            Pipeline Results
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-6 space-y-4">
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {/* Dataset Download */}
                            {results.dataset_download_url && (
                                <div className="rounded-lg border border-border bg-background p-4 space-y-3">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                        <Database className="w-4 h-4 text-primary" />
                                        Training Dataset
                                    </div>
                                    <p className="text-xs text-muted-foreground">Clean JSONL file ready for fine-tuning</p>
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="w-full gap-1.5"
                                        onClick={() => window.open(results.dataset_download_url, '_blank')}
                                    >
                                        <Download className="w-3.5 h-3.5" />
                                        Download JSONL
                                    </Button>
                                </div>
                            )}

                            {/* Model Endpoint */}
                            {results.model_endpoint_url && (
                                <div className="rounded-lg border border-border bg-background p-4 space-y-3">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                        <Rocket className="w-4 h-4 text-primary" />
                                        Model Endpoint
                                    </div>
                                    <p className="text-xs text-muted-foreground font-mono break-all">{results.model_endpoint_url}</p>
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="w-full gap-1.5"
                                        onClick={() => navigator.clipboard.writeText(results.model_endpoint_url)}
                                    >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                        Copy Endpoint
                                    </Button>
                                </div>
                            )}

                            {/* Training Metrics */}
                            {results.training_metrics && (
                                <div className="rounded-lg border border-border bg-background p-4 space-y-3">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                        <BarChart3 className="w-4 h-4 text-primary" />
                                        Training Metrics
                                    </div>
                                    <div className="space-y-2">
                                        {results.training_metrics.duration_min && (
                                            <div className="flex items-center justify-between text-xs">
                                                <span className="text-muted-foreground flex items-center gap-1"><Timer className="w-3 h-3" /> Duration</span>
                                                <span className="font-medium">{results.training_metrics.duration_min} min</span>
                                            </div>
                                        )}
                                        {results.training_metrics.final_loss != null && (
                                            <div className="flex items-center justify-between text-xs">
                                                <span className="text-muted-foreground">Final Loss</span>
                                                <span className="font-medium">{results.training_metrics.final_loss}</span>
                                            </div>
                                        )}
                                        {results.training_metrics.job_name && (
                                            <div className="flex items-center justify-between text-xs">
                                                <span className="text-muted-foreground">Job</span>
                                                <span className="font-medium truncate max-w-[150px]">{results.training_metrics.job_name}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Virtual Mind Agents */}
                            {results.virtual_mind_agents && (
                                <div className="rounded-lg border border-border bg-background p-4 space-y-3">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                        <Bot className="w-4 h-4 text-primary" />
                                        Deployed Agents
                                    </div>
                                    <div className="space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar">
                                        {results.virtual_mind_agents.map((agent, i) => (
                                            <div key={i} className="flex flex-col gap-2 text-xs border border-border/50 rounded-md p-2 bg-muted/20">
                                                <div>
                                                    <span className="font-semibold text-foreground block">{agent.name}</span>
                                                    {agent.role && <span className="text-muted-foreground/80">{agent.role}</span>}
                                                </div>
                                                <Button
                                                    size="sm"
                                                    variant="secondary"
                                                    className="w-full gap-1.5 h-7 text-[10px]"
                                                    onClick={() => window.open(agent.endpoint || 'http://localhost:8000', '_blank')}
                                                >
                                                    <ExternalLink className="w-3 h-3" />
                                                    Chat
                                                </Button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Automations */}
                            {results.virtual_mind_automations && (
                                <div className="rounded-lg border border-border bg-background p-4 space-y-3">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                        <Zap className="w-4 h-4 text-primary" />
                                        n8n Automations
                                    </div>
                                    <div className="space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar">
                                        {results.virtual_mind_automations.map((auto, i) => (
                                            <AutomationCard key={i} auto={auto} />
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Step Results Panel */}
            {results?.step_results && Object.keys(results.step_results).length > 0 && (
                <Card className="border-border bg-card mt-6">
                    <CardHeader className="pb-3 border-b border-border bg-muted/20">
                        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-muted-foreground uppercase tracking-wider">
                            <Layers className="w-4 h-4" />
                            Pipeline Step Results
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                        <div className="divide-y divide-border">
                            {Object.entries(results.step_results).map(([stepName, data]) => {
                                const summaryLines = buildStepSummary(stepName, data)
                                if (!summaryLines.length) return null
                                const hasError = data?.errors?.length > 0 || data?.files_failed > 0
                                return (
                                    <div key={stepName} className="px-4 py-3 flex flex-col sm:flex-row sm:items-start gap-2">
                                        <div className="flex items-center gap-2 min-w-[140px] shrink-0">
                                            <div className={`w-2 h-2 rounded-full ${hasError ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                                            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                                {stepName.replace(/_/g, ' ')}
                                            </span>
                                        </div>
                                        <div className="text-xs text-foreground space-y-0.5">
                                            {summaryLines.map((line, i) => (
                                                <p key={i} className={line.startsWith('⚠') || line.startsWith('❌') ? 'text-amber-400' : ''}>
                                                    {line}
                                                </p>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </CardContent>
                </Card>
            )}
            {/* ── Logs Panel ─────────────────────────────────────────────────── */}
            <div className="flex items-center justify-between pt-4">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                    <Terminal className="w-4 h-4" />
                    Execution Logs
                </h2>
                <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => setShowLogs(!showLogs)}
                    className="text-xs gap-1.5"
                >
                    {showLogs ? 'Hide Logs' : 'Show Logs'}
                    {fetchingLogs && <Loader2 className="w-3 h-3 animate-spin" />}
                </Button>
            </div>

            {showLogs && (
                <Card className="border-border bg-slate-950 shadow-2xl overflow-hidden">
                    <CardContent className="p-0">
                        <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
                            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">System Events</span>
                            <div className="flex items-center gap-2">
                                {logError && <span className="text-[10px] text-amber-400 font-mono">⚠ polling error</span>}
                                <span className="text-[10px] font-mono text-slate-400">{logs.length} events logged</span>
                            </div>
                        </div>
                        <div className="max-h-[350px] overflow-y-auto p-2 font-mono text-[11px] space-y-0.5 custom-scrollbar">
                            {logs.length === 0 ? (
                                <div className="py-10 text-center text-slate-500 italic">
                                    Waiting for execution data...
                                </div>
                            ) : (
                                displayLogs.map(entry => {
                                    const statusColors = {
                                        error: 'text-red-400',
                                        success: 'text-emerald-400',
                                        running: 'text-blue-400',
                                        info: 'text-slate-400'
                                    }

                                    return (
                                        <div key={entry.id} className="group flex items-start gap-3 py-1 px-2 hover:bg-slate-800/50 rounded transition-colors border-l-2 border-transparent hover:border-primary/30">
                                            <span className="text-slate-500 shrink-0 tabular-nums">
                                                {new Date(entry.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                            </span>
                                            <span className={`shrink-0 font-bold w-4 text-center ${statusColors[entry.status]}`}>
                                                {entry.icon}
                                            </span>
                                            <div className="flex-1 min-w-0">
                                                <p className={`font-semibold ${statusColors[entry.status]}`}>{entry.message}</p>
                                                {entry.detail && (
                                                    <p className="text-slate-300 mt-0.5 line-clamp-1 group-hover:line-clamp-none transition-all">
                                                        {entry.detail}
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    )
                                })
                            )}
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}
