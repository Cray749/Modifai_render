import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { Clock, FileText, ArrowRight } from 'lucide-react'

const statusConfig = {
    running: { label: 'Running', className: 'bg-blue-500/15 text-blue-400 border-blue-500/30' },
    completed: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
    complete: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
    failed: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30' },
    error: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30' },
    pending: { label: 'Pending', className: 'bg-amber-500/15 text-amber-400 border-amber-500/30' },
}

export default function ProjectCard({ project }) {
    const navigate = useNavigate()
    const status = statusConfig[project.status] || statusConfig.pending
    const pipeline = project.pipeline || []
    const completedSteps = pipeline.filter(s => s.status === 'complete').length
    const totalSteps = pipeline.length || 7
    const progressPercent = pipeline.length > 0 ? (completedSteps / totalSteps) * 100
        : project.status === 'completed' || project.status === 'complete' ? 100
            : project.status === 'running' ? 50
                : 0

    const timeAgo = (dateStr) => {
        if (!dateStr) return 'Just now'
        const diff = Date.now() - new Date(dateStr).getTime()
        const days = Math.floor(diff / (1000 * 60 * 60 * 24))
        if (days === 0) return 'Today'
        if (days === 1) return 'Yesterday'
        return `${days}d ago`
    }

    return (
        <div
            onClick={() => navigate(`/projects/${project.id}`)}
            className="group relative rounded-xl border border-border bg-card p-5 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:shadow-primary/5 overflow-hidden"
        >
            {/* Top: name + badge */}
            <div className="flex items-start justify-between gap-3 mb-3">
                <h3 className="font-semibold text-sm leading-tight line-clamp-1 group-hover:text-primary transition-colors">
                    {project.name}
                </h3>
                <Badge variant="outline" className={`shrink-0 text-[11px] px-2 py-0.5 ${status.className}`}>
                    {status.label}
                </Badge>
            </div>

            {/* Description */}
            <p className="text-xs text-muted-foreground line-clamp-2 mb-4">
                {project.description}
            </p>

            {/* Pipeline progress bar */}
            <div className="mb-3">
                <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] text-muted-foreground font-medium">
                        {project.mode ? project.mode.replace(/_/g, ' ') : 'Pipeline'}
                    </span>
                    <span className="text-[11px] text-muted-foreground">
                        {Math.round(progressPercent)}%
                    </span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-500 ${project.status === 'error' || project.status === 'failed'
                            ? 'bg-red-500'
                            : project.status === 'complete' || project.status === 'completed'
                                ? 'bg-emerald-500'
                                : 'bg-primary'
                            }`}
                        style={{ width: `${progressPercent}%` }}
                    />
                </div>
            </div>

            {/* Footer: meta */}
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {timeAgo(project.createdAt)}
                    </span>
                </div>
                <ArrowRight className="w-3.5 h-3.5 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 text-primary" />
            </div>
        </div>
    )
}

