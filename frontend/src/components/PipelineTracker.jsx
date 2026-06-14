import React from 'react'
import {
    Upload,
    ScanText,
    Layers,
    Database,
    ShieldCheck,
    Brain,
    Rocket,
    Check,
    Loader2,
    AlertCircle,
    Circle,
} from 'lucide-react'

const iconMap = { Upload, ScanText, Layers, Database, ShieldCheck, Brain, Rocket }

const statusStyles = {
    complete: {
        circle: 'bg-emerald-500 border-emerald-500 text-canvas',
        label: 'text-emerald-400',
        line: 'bg-emerald-500',
        StatusIcon: Check,
    },
    running: {
        circle: 'bg-primary/20 border-primary text-primary animate-pulse',
        label: 'text-primary',
        line: 'bg-primary/40',
        StatusIcon: Loader2,
    },
    error: {
        circle: 'bg-red-500/20 border-red-500 text-red-400',
        label: 'text-red-400',
        line: 'bg-red-500/40',
        StatusIcon: AlertCircle,
    },
    pending: {
        circle: 'bg-muted border-border text-muted-foreground',
        label: 'text-muted-foreground',
        line: 'bg-border',
        StatusIcon: Circle,
    },
}

export default function PipelineTracker({ steps, pipeline, activeStep, onStepClick }) {
    return (
        <div className="w-full">
            {/* Horizontal tracker for md+ */}
            <div className="hidden md:flex items-start justify-between gap-0">
                {steps.map((step, i) => {
                    const Icon = iconMap[step.icon] || Circle
                    const pStep = pipeline[i]
                    const style = statusStyles[pStep.status]
                    const isActive = activeStep === i
                    const isLast = i === steps.length - 1

                    return (
                        <div key={step.id} className="flex items-start flex-1 last:flex-none">
                            <button
                                onClick={() => onStepClick(i)}
                                className={`flex flex-col items-center gap-2 group transition-all duration-200 ${isActive ? 'scale-105' : 'hover:scale-105'
                                    }`}
                            >
                                {/* Circle */}
                                <div className={`
                                    relative w-11 h-11 rounded-full border-2 flex items-center justify-center
                                    transition-all duration-300 shrink-0
                                    ${style.circle}
                                    ${isActive ? 'ring-2 ring-primary/30 ring-offset-2 ring-offset-background' : ''}
                                `}>
                                    {pStep.status === 'complete' ? (
                                        <Check className="w-5 h-5" />
                                    ) : pStep.status === 'running' ? (
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                    ) : pStep.status === 'error' ? (
                                        <AlertCircle className="w-5 h-5" />
                                    ) : (
                                        <Icon className="w-4 h-4" />
                                    )}
                                </div>

                                {/* Label */}
                                <span className={`text-[11px] font-medium text-center w-16 leading-tight ${style.label}`}>
                                    {step.name}
                                </span>

                                {/* Progress (for running steps) */}
                                {pStep.status === 'running' && (
                                    <span className="text-[10px] text-primary font-mono">{pStep.progress}%</span>
                                )}
                            </button>

                            {/* Connector line */}
                            {!isLast && (
                                <div className="flex-1 mt-5 mx-1">
                                    <div className="h-0.5 w-full bg-border rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${style.line}`}
                                            style={{
                                                width: pStep.status === 'complete' ? '100%'
                                                    : pStep.status === 'running' ? `${pStep.progress}%`
                                                        : '0%'
                                            }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Vertical tracker for mobile */}
            <div className="md:hidden space-y-0">
                {steps.map((step, i) => {
                    const Icon = iconMap[step.icon] || Circle
                    const pStep = pipeline[i]
                    const style = statusStyles[pStep.status]
                    const isActive = activeStep === i
                    const isLast = i === steps.length - 1

                    return (
                        <div key={step.id} className="flex gap-3">
                            {/* Circle + line */}
                            <div className="flex flex-col items-center">
                                <button
                                    onClick={() => onStepClick(i)}
                                    className={`
                                        w-9 h-9 rounded-full border-2 flex items-center justify-center shrink-0
                                        transition-all duration-300
                                        ${style.circle}
                                        ${isActive ? 'ring-2 ring-primary/30 ring-offset-2 ring-offset-background' : ''}
                                    `}
                                >
                                    {pStep.status === 'complete' ? (
                                        <Check className="w-4 h-4" />
                                    ) : pStep.status === 'running' ? (
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : pStep.status === 'error' ? (
                                        <AlertCircle className="w-4 h-4" />
                                    ) : (
                                        <Icon className="w-3.5 h-3.5" />
                                    )}
                                </button>
                                {!isLast && (
                                    <div className="w-0.5 h-8 bg-border mt-1">
                                        <div
                                            className={`w-full rounded-full transition-all duration-500 ${style.line}`}
                                            style={{
                                                height: pStep.status === 'complete' ? '100%'
                                                    : pStep.status === 'running' ? `${pStep.progress}%`
                                                        : '0%',
                                            }}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* Text */}
                            <div className="pb-6">
                                <p className={`text-sm font-medium ${style.label}`}>{step.name}</p>
                                <p className="text-xs text-muted-foreground">{step.description}</p>
                                {pStep.status === 'running' && (
                                    <span className="text-[11px] text-primary font-mono mt-0.5 inline-block">{pStep.progress}%</span>
                                )}
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
