import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    FileText,
    Upload,
    Settings2,
    CheckCircle2,
    ArrowLeft,
    ArrowRight,
    Sparkles,
    Check,
    Target,
    Layers,
    MessageSquare,
    BookOpen,
    Palette,
    Tags,
    Bot,
    Zap,
    Database,
    Rocket,
    Brain,
    AlertCircle,
    ChevronLeft,
    Server
} from 'lucide-react'
import FileUploadZone from '@/components/FileUploadZone'
import SpotlightCard from '@/components/ui/SpotlightCard'
import { apiClient } from '@/api/client'
import { extractTextSample, pickRandomFiles } from '@/utils/textExtractor'

// ── Mode Definitions ──────────────────────────────────────────────────────────
const MODES = [
    {
        value: 'dataset_only',
        label: 'Generate Dataset',
        icon: Database,
        desc: 'Create training-ready datasets from documents.',
    },
    {
        value: 'finetune_only',
        label: 'Fine-Tune Model',
        icon: Brain,
        desc: 'Build an AI model tailored to your domain.',
    },
    {
        value: 'deploy_agents',
        label: 'Deploy Agents',
        icon: Server,
        desc: 'Create specialized AI agents in minutes.',
    },
    {
        value: 'generate_automations',
        label: 'Generate Automations',
        icon: Zap,
        desc: 'Transform processes into automated workflows.',
    },
    {
        value: 'full',
        label: 'Full Pipeline',
        icon: Layers,
        desc: 'Build your complete AI ecosystem.',
    },
]

// ── Intent Definitions ────────────────────────────────────────────────────────
const INTENTS = [
    {
        value: 'question-answering',
        label: 'Question Answering',
        icon: MessageSquare,
        desc: 'Train a model to answer questions about your domain knowledge.',
    },
    {
        value: 'summarization',
        label: 'Summarization',
        icon: BookOpen,
        desc: 'Train a model to summarize long documents into concise outputs.',
    },
    {
        value: 'tone-rewriting',
        label: 'Tone Rewriting',
        icon: Palette,
        desc: 'Train a model to rewrite text in a specific tone or style.',
    },
    {
        value: 'classification',
        label: 'Classification',
        icon: Tags,
        desc: 'Train a model to categorize text into predefined labels.',
    },
    {
        value: 'general-assistant',
        label: 'General Assistant',
        icon: Bot,
        desc: 'Train a general-purpose assistant for your domain.',
    },
]

// ── Model Definitions ─────────────────────────────────────────────────────────
const MODELS = [
    { value: 'llama-3.1-8b', label: 'Llama 3.1 8B', desc: 'Fast, efficient general-purpose' },
    { value: 'mistral-7b', label: 'Mistral 7B', desc: 'Strong reasoning and instruction' },
    { value: 'gemma-2-9b', label: 'Gemma 2 9B', desc: 'Balanced performance' },
    { value: 'phi-3-mini', label: 'Phi-3 Mini', desc: 'Compact, great for edge' },
]

// ── Dynamic Steps Based on Mode ───────────────────────────────────────────────
function getSteps(mode) {
    const isVirtualMind = mode === 'deploy_agents' || mode === 'generate_automations'
    const isFinetuneOnly = mode === 'finetune_only'

    const steps = [
        { id: 'mode', label: 'Pipeline Mode', icon: Layers },
    ]

    if (!isFinetuneOnly) {
        steps.push({ id: 'intent', label: 'Intent', icon: Target })
    }

    steps.push({ id: 'upload', label: 'Upload Data', icon: Upload })

    if (!isFinetuneOnly) {
        steps.push({ id: 'evaluate', label: 'Evaluate', icon: Sparkles })
    }

    if (!isVirtualMind) {
        steps.push({ id: 'config', label: 'Configuration', icon: Settings2 })
    }

    steps.push({ id: 'review', label: 'Review & Start', icon: CheckCircle2 })

    return steps
}

export default function NewProjectPage() {
    const navigate = useNavigate()
    const [currentStep, setCurrentStep] = useState(0)

    // Form state
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [mode, setMode] = useState('dataset_only')
    const [intent, setIntent] = useState('')
    const [useCase, setUseCase] = useState('')
    const [files, setFiles] = useState([])
    const [model, setModel] = useState('llama-3.1-8b')
    const [samplesPerChunk, setSamplesPerChunk] = useState(5)
    const [qualityThreshold, setQualityThreshold] = useState(70)

    // Evaluate Data Phase
    const [evaluating, setEvaluating] = useState(false)
    const [evalScore, setEvalScore] = useState(null)
    const [evalExplanation, setEvalExplanation] = useState(null)
    const [extractedText, setExtractedText] = useState('')

    // Submission state
    const [submitting, setSubmitting] = useState(false)
    const [submitError, setSubmitError] = useState(null)

    const steps = getSteps(mode)
    const currentStepId = steps[currentStep]?.id

    const isVirtualMind = mode === 'deploy_agents' || mode === 'generate_automations'
    const requiresIntentAndEval = mode !== 'finetune_only'
    const showDatasetConfig = mode === 'dataset_only' || mode === 'dataset_and_finetune' || mode === 'full'
    const showFinetuneConfig = mode === 'finetune_only' || mode === 'dataset_and_finetune' || mode === 'full'

    const canProceed = () => {
        switch (currentStepId) {
            case 'mode': return name.trim().length >= 3
            case 'intent': return intent && useCase.trim().length >= 10
            case 'upload': return files.length > 0
            case 'evaluate': return evalScore !== null || (evalExplanation !== null && !evaluating)
            case 'config': return !!model
            case 'review': return true
            default: return false
        }
    }

    const handleModeChange = (newMode) => {
        setMode(newMode)
        setCurrentStep(0) // Always go back to step 0 when mode changes
        // Reset evaluation
        setEvalScore(null)
        setEvalExplanation(null)
        setExtractedText('')
        // Reset intent if switching to finetune_only
        if (newMode === 'finetune_only') {
            setIntent('')
            setUseCase('')
        }
    }

    const handleEvaluate = async () => {
        if (!files.length) return

        setEvaluating(true)
        setEvalScore(null)
        setEvalExplanation(null)

        try {
            // Pick up to 3 random text-extractable files
            const sampled = pickRandomFiles(files, 3)

            const results = []
            const previews = []

            for (const fileObj of sampled) {
                try {
                    const text = await extractTextSample(fileObj.file)
                    previews.push(`[${fileObj.name}] ${text.substring(0, 100)}...`)

                    const result = await apiClient.post('evaluate', {
                        text_sample: text,
                        intent: `${intent}: ${useCase}`,
                    })
                    results.push({ name: fileObj.name, score: result.score, explanation: result.explanation })
                } catch (fileErr) {
                    // Skip files that fail extraction, log for debugging
                    console.warn(`Skipped ${fileObj.name}:`, fileErr.message)
                }
            }

            if (results.length === 0) {
                throw new Error('Could not evaluate any of the sampled files. Try different file formats (.txt, .pdf, .docx).')
            }

            // Average the scores across all evaluated files
            const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length

            // Build a combined explanation
            const explanation = results.length === 1
                ? results[0].explanation
                : `Averaged across ${results.length} sampled files (${results.map(r => `${r.name}: ${Math.round(r.score * 100)}%`).join(', ')}). ${results[results.length - 1].explanation}`

            setEvalScore(avgScore)
            setEvalExplanation(explanation)
            setExtractedText(previews.join('\n\n'))
        } catch (error) {
            console.error("Evaluation error:", error)
            setEvalExplanation(error.message || "Failed to evaluate the dataset. You can proceed anyway.")
            setEvalScore(null)
        } finally {
            setEvaluating(false)
        }
    }

    const handleSubmit = async () => {
        try {
            setSubmitting(true)
            setSubmitError(null)

            // 1. Create the project
            const project = await apiClient.post('projects', {
                name,
                description,
                mode,
                intent: requiresIntentAndEval ? intent : null,
                base_model: 'gpt-4o-mini',
                config: {},
            })

            // 2. Upload files via presigned URLs
            for (const fileObj of files) {
                const urlData = await apiClient.post(
                    `projects/${project.id}/upload-url?filename=${encodeURIComponent(fileObj.name)}&content_type=${encodeURIComponent(fileObj.type || 'application/octet-stream')}`
                )
                const uploadResponse = await fetch(urlData.presigned_url, {
                    method: 'PUT',
                    body: fileObj.file,
                    headers: { 'Content-Type': fileObj.type || 'application/octet-stream' },
                })
                if (!uploadResponse.ok) throw new Error(`Failed to upload ${fileObj.name}`)
            }

            // 3. Start the pipeline
            await apiClient.post(`projects/${project.id}/start`, {
                config: {
                    intent: requiresIntentAndEval ? `${intent}: ${useCase}` : undefined,
                },
                uploaded_filenames: files.map(f => f.name),
            })

            // 4. Navigate to the detail page
            navigate(`/projects/${project.id}`)

        } catch (error) {
            console.error("Submission error:", error)
            setSubmitError(error.message || 'Failed to create project')
        } finally {
            setSubmitting(false)
        }
    }

    return (
        <div className="min-h-screen bg-canvas-soft font-sans selection:bg-ink selection:text-canvas pb-24">
            <div className="max-w-[800px] mx-auto px-4 md:px-6 py-12 md:py-16 space-y-8">
                
                {/* Header */}
                <div className="flex items-start gap-4">
                    <button 
                        onClick={() => navigate(-1)}
                        className="w-10 h-10 rounded-full border border-hairline bg-canvas flex items-center justify-center text-body hover:text-ink hover:border-hairline-strong transition-colors shrink-0 mt-1"
                    >
                        <ArrowLeft className="w-5 h-5" strokeWidth={1.5} />
                    </button>
                    <div>
                        <h1 className="text-[32px] leading-[40px] font-semibold tracking-[-1.28px] text-ink">
                            Create New Project
                        </h1>
                        <p className="text-[16px] leading-[24px] text-body mt-1">
                            Set up your data-to-model pipeline.
                        </p>
                    </div>
                </div>

                {/* Vercel-style Stepper */}
                <div className="flex items-center gap-0 overflow-x-auto pb-4 scrollbar-hide">
                    {steps.map((step, i) => {
                        const isCompleted = i < currentStep
                        const isActive = i === currentStep
                        const isFuture = i > currentStep

                        return (
                            <React.Fragment key={step.id}>
                                <button
                                    onClick={() => i < currentStep && setCurrentStep(i)}
                                    disabled={isFuture}
                                    className={`
                                        flex flex-col gap-2 relative transition-all duration-200 group shrink-0
                                        ${isActive ? 'cursor-default' : isCompleted ? 'cursor-pointer' : 'cursor-default opacity-50'}
                                    `}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`
                                            w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border transition-colors
                                            ${isActive ? 'bg-ink border-ink text-canvas' : 
                                              isCompleted ? 'bg-ink border-ink text-canvas' : 
                                              'bg-canvas border-hairline text-mute'}
                                        `}>
                                            {isCompleted ? <Check className="w-3.5 h-3.5" strokeWidth={2.5} /> : i + 1}
                                        </div>
                                        <span className={`font-mono text-[12px] uppercase tracking-wider ${isActive ? 'text-ink font-semibold' : 'text-mute'}`}>
                                            {step.label}
                                        </span>
                                    </div>
                                </button>
                                {i < steps.length - 1 && (
                                    <div className={`w-8 sm:flex-1 h-[1px] mx-4 transition-colors duration-300 ${isCompleted ? 'bg-ink' : 'bg-hairline'}`} />
                                )}
                            </React.Fragment>
                        )
                    })}
                </div>

                {/* Step Content Container */}
                <div className="bg-canvas rounded-[12px] border border-hairline shadow-[0px_1px_1px_#00000005,0px_2px_2px_#0000000a] p-8 md:p-10">
                    
                    {/* ── Step 1: Mode Selection ── */}
                    {currentStepId === 'mode' && (
                        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <div className="space-y-6">
                                <div className="space-y-2">
                                    <label htmlFor="name" className="block text-[14px] font-medium text-ink">Project Name</label>
                                    <input
                                        id="name"
                                        type="text"
                                        placeholder="e.g. Customer Support Bot"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        className="w-full h-10 px-3 bg-canvas border border-hairline rounded-[6px] text-[14px] text-ink focus:outline-none focus:ring-1 focus:ring-[#171717] focus:border-ink transition-all placeholder:text-mute"
                                    />
                                    {name.length > 0 && name.length < 3 && (
                                        <p className="text-[12px] text-[#ee0000]">Name must be at least 3 characters</p>
                                    )}
                                </div>

                                <div className="space-y-2">
                                    <label htmlFor="desc" className="block text-[14px] font-medium text-ink">Description <span className="text-mute font-normal">(Optional)</span></label>
                                    <textarea
                                        id="desc"
                                        placeholder="Briefly describe your project"
                                        value={description}
                                        onChange={(e) => setDescription(e.target.value)}
                                        rows={2}
                                        className="w-full p-3 bg-canvas border border-hairline rounded-[6px] text-[14px] text-ink focus:outline-none focus:ring-1 focus:ring-[#171717] focus:border-ink transition-all resize-none placeholder:text-mute"
                                    />
                                </div>
                            </div>

                            <div className="w-full h-[1px] bg-hairline" />

                            <div className="space-y-4">
                                <label className="block text-[14px] font-medium text-ink">Pipeline Mode</label>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {MODES.map((m) => {
                                        const ModeIcon = m.icon
                                        const isSelected = mode === m.value
                                        return (
                                            <SpotlightCard
                                                key={m.value}
                                                className={`
                                                    relative flex cursor-pointer rounded-[8px] border transition-all
                                                    ${isSelected
                                                        ? 'border-ink bg-canvas-soft ring-1 ring-[#171717]'
                                                        : 'border-hairline bg-canvas hover:border-hairline-strong'
                                                    }
                                                `}
                                            >
                                                <label className="w-full p-5 flex cursor-pointer">
                                                    <input
                                                        type="radio"
                                                        name="mode"
                                                        value={m.value}
                                                        className="sr-only"
                                                        onChange={() => handleModeChange(m.value)}
                                                        checked={isSelected}
                                                    />
                                                    <div className="flex items-start gap-4">
                                                        <div className={`w-10 h-10 rounded-full border flex items-center justify-center shrink-0 transition-colors ${isSelected ? 'bg-ink border-ink text-canvas' : 'bg-canvas-soft border-hairline text-ink'}`}>
                                                            <ModeIcon className="w-4.5 h-4.5" strokeWidth={1.5} />
                                                        </div>
                                                        <div className="flex flex-col gap-1">
                                                            <span className="block text-[14px] font-medium text-ink">{m.label}</span>
                                                            <span className="block text-[14px] leading-[20px] text-body">{m.desc}</span>
                                                        </div>
                                                    </div>
                                                </label>
                                            </SpotlightCard>
                                        )
                                    })}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ── Step 2: Intent Selection ── */}
                    {currentStepId === 'intent' && (
                        <div className="space-y-8 animate-in slide-in-from-right-8 duration-500">
                            <div className="space-y-4">
                                <label className="block text-[14px] font-medium text-ink">Intent</label>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {INTENTS.map((item) => {
                                        const IntentIcon = item.icon
                                        const isSelected = intent === item.value
                                        return (
                                            <SpotlightCard
                                                key={item.value}
                                                className={`
                                                    relative flex flex-col items-start cursor-pointer rounded-[8px] border transition-all
                                                    ${isSelected
                                                        ? 'border-ink bg-canvas-soft ring-1 ring-[#171717]'
                                                        : 'border-hairline bg-canvas hover:border-hairline-strong'
                                                    }
                                                `}
                                            >
                                                <label className="w-full p-5 flex flex-col items-start cursor-pointer gap-4">
                                                    <input
                                                        type="radio"
                                                        name="intent"
                                                        value={item.value}
                                                        className="sr-only"
                                                        onChange={() => setIntent(item.value)}
                                                        checked={isSelected}
                                                    />
                                                    <div className={`w-10 h-10 rounded-full border flex items-center justify-center transition-colors ${isSelected ? 'bg-ink border-ink text-canvas' : 'bg-canvas-soft border-hairline text-ink'}`}>
                                                        <IntentIcon className="w-5 h-5" strokeWidth={1.5} />
                                                    </div>
                                                    <div>
                                                        <span className="block text-[14px] font-medium text-ink mb-1">{item.label}</span>
                                                        <span className="block text-[12px] leading-[18px] text-body">{item.desc}</span>
                                                    </div>
                                                </label>
                                            </SpotlightCard>
                                        )
                                    })}
                                </div>
                            </div>

                            <div className="w-full h-[1px] bg-hairline" />

                            <div className="space-y-2">
                                <label htmlFor="usecase" className="block text-[14px] font-medium text-ink">Use-Case Description</label>
                                <textarea
                                    id="usecase"
                                    placeholder="Describe how you intend to use the fine-tuned model. This guides the LLM to generate relevant training examples."
                                    value={useCase}
                                    onChange={(e) => setUseCase(e.target.value)}
                                    rows={3}
                                    className="w-full p-3 bg-canvas border border-hairline rounded-[6px] text-[14px] text-ink focus:outline-none focus:ring-1 focus:ring-[#171717] focus:border-ink transition-all resize-none placeholder:text-mute"
                                />
                                {useCase.length > 0 && useCase.length < 10 && (
                                    <p className="text-[12px] text-[#ee0000]">Description must be at least 10 characters</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* ── Step 3: Upload Files ── */}
                    {currentStepId === 'upload' && (
                        <div className="animate-in slide-in-from-right-8 duration-500">
                            <div className="mb-6">
                                <h2 className="text-[18px] font-semibold text-ink">Upload Your Data</h2>
                                <p className="text-[14px] text-body">
                                    {mode === 'finetune_only'
                                        ? 'Upload your JSONL training dataset file.'
                                        : 'Upload the documents you want to use for training data generation.'
                                    }
                                </p>
                            </div>
                            <FileUploadZone files={files} onFilesChange={setFiles} />
                        </div>
                    )}

                    {/* ── Step 4: Data Evaluation ── */}
                    {currentStepId === 'evaluate' && (
                        <div className="space-y-8 animate-in slide-in-from-right-8 duration-500">
                            <div className="space-y-1">
                                <h2 className="text-[18px] font-semibold text-ink flex items-center gap-2">
                                    <Sparkles className="w-5 h-5 text-ink" />
                                    Data Quality Check
                                </h2>
                                <p className="text-[14px] text-body">
                                    We'll randomly sample from your files and evaluate if the content is suitable for <span className="font-medium text-ink">"{intent}"</span>.
                                </p>
                            </div>

                            {!evalScore && !evaluating && !evalExplanation && (
                                <div className="rounded-[8px] border border-dashed border-hairline p-12 text-center bg-canvas-soft">
                                    <div className="mx-auto w-12 h-12 border border-hairline bg-canvas rounded-full flex items-center justify-center mb-4">
                                        <FileText className="w-5 h-5 text-ink" strokeWidth={1.5} />
                                    </div>
                                    <h3 className="text-[16px] font-semibold text-ink mb-2">Ready to Test</h3>
                                    <p className="text-[14px] text-body mb-6 max-w-sm mx-auto">
                                        We'll pick up to 3 random files, extract text from random sections, and ask Nova Micro to evaluate quality.
                                    </p>
                                    <button 
                                        onClick={handleEvaluate} 
                                        className="bg-ink text-canvas px-6 h-10 rounded-full text-[14px] font-medium hover:bg-ink/90 transition-colors flex items-center gap-2 mx-auto"
                                    >
                                        <Sparkles className="w-4 h-4" />
                                        Run Automated Evaluation
                                    </button>
                                </div>
                            )}

                            {!evalScore && !evaluating && evalExplanation && (
                                <div className="rounded-[8px] border border-[#f7d4d6] bg-[#f7d4d6]/30 p-6 space-y-4">
                                    <div className="flex items-start gap-3">
                                        <AlertCircle className="w-5 h-5 text-[#ee0000] shrink-0 mt-0.5" />
                                        <div className="space-y-1">
                                            <h3 className="font-semibold text-[#ee0000]">Evaluation Failed</h3>
                                            <p className="text-[14px] text-body leading-relaxed">{evalExplanation}</p>
                                        </div>
                                    </div>
                                    <div className="flex gap-3">
                                        <button 
                                            onClick={handleEvaluate} 
                                            className="bg-canvas border border-hairline text-ink px-4 h-8 rounded-full text-[12px] font-medium hover:bg-canvas-soft transition-colors flex items-center gap-2"
                                        >
                                            <Sparkles className="w-3.5 h-3.5" />
                                            Retry
                                        </button>
                                        <button 
                                            onClick={() => setCurrentStep(currentStep + 1)} 
                                            className="text-[12px] font-medium text-mute hover:text-ink transition-colors"
                                        >
                                            Skip evaluation →
                                        </button>
                                    </div>
                                </div>
                            )}

                            {evaluating && (
                                <div className="rounded-[8px] border border-hairline p-12 text-center bg-canvas shadow-[0px_1px_1px_#00000005]">
                                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-2 border-hairline border-t-[#171717] mb-6" />
                                    <p className="text-[14px] font-medium text-ink animate-pulse">Sampling random files & analyzing quality...</p>
                                </div>
                            )}

                            {evalScore !== null && !evaluating && (
                                <div className="space-y-6">
                                    <div className={`rounded-[8px] border p-8 transition-all shadow-[0px_1px_1px_#00000005] bg-canvas ${
                                        evalScore > 0.6 ? 'border-hairline' :
                                        evalScore > 0.4 ? 'border-[#f5a623]' : 'border-[#ee0000]'
                                    }`}>
                                        <div className="flex flex-col sm:flex-row items-center gap-8">
                                            <div className="relative shrink-0">
                                                <svg className="w-24 h-24 transform -rotate-90">
                                                    <circle cx="48" cy="48" r="40" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-[#fafafa] border-hairline" />
                                                    <circle cx="48" cy="48" r="40" stroke="currentColor" strokeWidth="8" fill="transparent"
                                                        strokeDasharray={40 * 2 * Math.PI}
                                                        strokeDashoffset={40 * 2 * Math.PI * (1 - evalScore)}
                                                        className={`transition-all duration-1000 ease-out ${
                                                            evalScore > 0.6 ? 'text-[#0070f3]' :
                                                            evalScore > 0.4 ? 'text-[#f5a623]' : 'text-[#ee0000]'
                                                        }`}
                                                    />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <span className="text-[20px] font-semibold text-ink">{Math.round(evalScore * 100)}%</span>
                                                </div>
                                            </div>
                                            <div className="space-y-3 text-center sm:text-left">
                                                <h3 className="font-semibold text-[18px] text-ink flex items-center justify-center sm:justify-start gap-2">
                                                    {evalScore > 0.6 && <CheckCircle2 className="w-5 h-5 text-[#0070f3]" />}
                                                    {evalScore > 0.6 ? 'High Quality' : evalScore > 0.4 ? 'Acceptable Quality' : 'Low Quality'}
                                                </h3>
                                                <p className="text-[14px] text-body leading-relaxed">
                                                    {evalExplanation}
                                                </p>
                                                {evalScore < 0.6 && (
                                                    <button 
                                                        onClick={handleEvaluate} 
                                                        className="mt-2 bg-canvas border border-hairline text-ink px-4 h-8 rounded-full text-[12px] font-medium hover:bg-canvas-soft transition-colors"
                                                    >
                                                        Retest another sample
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="rounded-[8px] border border-hairline bg-canvas-soft p-6">
                                        <p className="font-mono text-[10px] uppercase tracking-wider text-mute mb-3">Random Samples Extracted & Tested</p>
                                        <p className="font-mono text-[12px] text-body leading-relaxed bg-canvas p-4 rounded-[6px] border border-hairline whitespace-pre-line max-h-64 overflow-y-auto">
                                            {extractedText}
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* ── Step 5: Configuration ── */}
                    {currentStepId === 'config' && (
                        <div className="space-y-8 animate-in slide-in-from-right-8 duration-500">
                            <div className="space-y-1">
                                <h2 className="text-[18px] font-semibold text-ink">Pipeline Processing Overview</h2>
                                <p className="text-[14px] text-body">Based on your uploaded documents, here is the projected pipeline execution plan.</p>
                            </div>

                            {(() => {
                                const totalBytes = files.reduce((acc, f) => acc + (f.size || 0), 0);
                                const estChars = Math.max(100, Math.floor(totalBytes / 2));
                                const estChunks = Math.max(1, Math.ceil(estChars / 12000));
                                const estMins = Math.max(1, Math.ceil(estChunks * 1.5));
                                return (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-2">
                                            <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Est. Chunks</p>
                                            <p className="text-[24px] font-semibold text-ink">{estChunks}</p>
                                            <p className="text-[12px] text-body">Dynamic chunks at ~12k chars</p>
                                        </div>
                                        <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-2">
                                            <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Samples / Chunk</p>
                                            <p className="text-[24px] font-semibold text-ink">1-3</p>
                                            <p className="text-[12px] text-body">Variable based on density</p>
                                        </div>
                                        <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-2">
                                            <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Est. Time</p>
                                            <p className="text-[24px] font-semibold text-ink">~{estMins}m</p>
                                            <p className="text-[12px] text-body">Pipeline completion time</p>
                                        </div>
                                    </div>
                                )
                            })()}
                        </div>
                    )}

                    {/* ── Step 6: Review & Start ── */}
                    {currentStepId === 'review' && (
                        <div className="space-y-8 animate-in slide-in-from-right-8 duration-500">
                            <div className="space-y-1">
                                <h2 className="text-[18px] font-semibold text-ink">Review & Start Pipeline</h2>
                                <p className="text-[14px] text-body">Verify your project settings before kicking off the pipeline</p>
                            </div>

                            <div className="space-y-4">
                                <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-3">
                                    <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Project</p>
                                    <p className="text-[16px] font-medium text-ink">{name}</p>
                                    {description && <p className="text-[14px] text-body">{description}</p>}
                                    <div className="inline-block mt-2 font-mono text-[10px] bg-canvas border border-hairline px-2 py-1 rounded-[4px] text-ink">
                                        {MODES.find(m => m.value === mode)?.label}
                                    </div>
                                </div>

                                {requiresIntentAndEval && intent && (
                                    <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-3">
                                        <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Intent</p>
                                        <p className="text-[16px] font-medium text-ink">{INTENTS.find(i => i.value === intent)?.label}</p>
                                        <p className="text-[14px] text-body">{useCase}</p>
                                    </div>
                                )}

                                <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-3">
                                    <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Files</p>
                                    <p className="text-[16px] font-medium text-ink">{files.length} file{files.length !== 1 ? 's' : ''}</p>
                                    <div className="flex flex-wrap gap-2 pt-1">
                                        {files.slice(0, 5).map(f => (
                                            <div key={f.id} className="font-mono text-[10px] bg-canvas border border-hairline px-2 py-1 rounded-[4px] text-body">
                                                {f.name}
                                            </div>
                                        ))}
                                        {files.length > 5 && <div className="font-mono text-[10px] bg-canvas border border-hairline px-2 py-1 rounded-[4px] text-body">+{files.length - 5} more</div>}
                                    </div>
                                </div>

                                <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-4">
                                    <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Configuration</p>
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
                                        {isVirtualMind && (
                                            <div className="col-span-full">
                                                <p className="text-[14px] text-body">Auto-managed by Virtual Mind Orchestrator</p>
                                            </div>
                                        )}
                                        {showFinetuneConfig && (
                                            <div>
                                                <p className="text-[12px] text-mute mb-1">Base Model</p>
                                                <p className="text-[14px] font-medium text-ink">{MODELS.find(m => m.value === model)?.label}</p>
                                            </div>
                                        )}
                                        {showDatasetConfig && (
                                            <>
                                                <div>
                                                    <p className="text-[12px] text-mute mb-1">Samples/Chunk</p>
                                                    <p className="text-[14px] font-medium text-ink">{samplesPerChunk}</p>
                                                </div>
                                                <div>
                                                    <p className="text-[12px] text-mute mb-1">Quality Threshold</p>
                                                    <p className="text-[14px] font-medium text-ink">{qualityThreshold}%</p>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>

                                {requiresIntentAndEval && evalScore !== null && (
                                    <div className="rounded-[8px] border border-hairline bg-canvas-soft p-5 space-y-3">
                                        <p className="font-mono text-[10px] uppercase tracking-wider text-mute">Data Quality</p>
                                        <p className="text-[16px] font-medium text-ink">{Math.round(evalScore * 100)}% — {evalScore > 0.6 ? 'High Quality' : evalScore > 0.4 ? 'Acceptable' : 'Low Quality'}</p>
                                    </div>
                                )}

                                {submitError && (
                                    <div className="rounded-[8px] border border-[#f7d4d6] bg-[#f7d4d6]/30 p-5 flex items-start gap-3">
                                        <AlertCircle className="w-5 h-5 text-[#ee0000] shrink-0 mt-0.5" />
                                        <div>
                                            <p className="text-[14px] font-medium text-[#ee0000]">Failed to start pipeline</p>
                                            <p className="text-[14px] text-body mt-1">{submitError}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* ── Navigation Buttons ── */}
                <div className="flex items-center justify-between pt-4">
                    <button
                        onClick={() => currentStep === 0 ? navigate(-1) : setCurrentStep(currentStep - 1)}
                        className="bg-canvas border border-hairline text-ink px-6 h-10 rounded-full text-[14px] font-medium hover:bg-canvas-soft transition-colors flex items-center gap-2 shadow-[0px_1px_1px_#00000005]"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        {currentStep === 0 ? 'Cancel' : 'Back'}
                    </button>

                    {currentStepId !== 'review' ? (
                        <button
                            disabled={!canProceed()}
                            onClick={() => setCurrentStep(currentStep + 1)}
                            className="bg-ink text-canvas px-6 h-10 rounded-full text-[14px] font-medium hover:bg-ink/90 transition-colors flex items-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Continue
                            <ArrowRight className="w-4 h-4" />
                        </button>
                    ) : (
                        <button
                            onClick={handleSubmit}
                            disabled={submitting}
                            className="bg-ink text-canvas px-8 h-10 rounded-full text-[14px] font-medium hover:bg-ink/90 transition-colors flex items-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {submitting ? (
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            ) : (
                                <Rocket className="w-4 h-4" />
                            )}
                            {submitting ? 'Starting...' : 'Start Pipeline'}
                        </button>
                    )}
                </div>
            </div>
        </div>
    )
}
