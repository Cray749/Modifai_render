import React from 'react'

export default function StatsCard({ icon: Icon, label, value, suffix = '', trend }) {
    return (
        <div className="group relative rounded-xl border border-border bg-card p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:shadow-primary/5 hover:border-primary/40">
            {/* Gradient glow on hover */}
            <div className="absolute -inset-px rounded-xl bg-gradient-to-br from-primary/20 via-transparent to-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10 blur-sm pointer-events-none" />

            <div className="flex items-start justify-between">
                <div className="space-y-2">
                    <p className="text-sm text-muted-foreground font-medium">{label}</p>
                    <div className="flex items-baseline gap-1">
                        <span className="text-3xl font-bold tracking-tight">{value}</span>
                        {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
                    </div>
                    {trend && (
                        <p className={`text-xs font-medium ${trend > 0 ? 'text-emerald-500' : 'text-red-400'}`}>
                            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from last month
                        </p>
                    )}
                </div>
                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10 shrink-0 transition-colors duration-300 group-hover:bg-primary/20">
                    <Icon className="w-5 h-5 text-primary" />
                </div>
            </div>
        </div>
    )
}
