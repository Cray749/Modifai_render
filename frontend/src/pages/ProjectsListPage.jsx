import React, { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    Search,
    Plus,
    LayoutGrid,
    List,
    ArrowRight,
    Clock,
    FileText,
    ArrowUpDown,
    Filter,
    Loader2,
    FolderOpen,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import ProjectCard from '@/components/ProjectCard'
import { apiClient } from '@/api/client'

const statusConfig = {
    running: { label: 'Running', className: 'bg-blue-500/15 text-blue-400 border-blue-500/30' },
    completed: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
    complete: { label: 'Complete', className: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' },
    failed: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30' },
    error: { label: 'Error', className: 'bg-red-500/15 text-red-400 border-red-500/30' },
    pending: { label: 'Pending', className: 'bg-amber-500/15 text-amber-400 border-amber-500/30' },
}

function timeAgo(dateStr) {
    if (!dateStr) return 'Just now'
    const diff = Date.now() - new Date(dateStr).getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    return `${days}d ago`
}

export default function ProjectsListPage() {
    const navigate = useNavigate()
    const [projects, setProjects] = useState([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [statusFilter, setStatusFilter] = useState('all')
    const [viewMode, setViewMode] = useState('grid')
    const [sortBy, setSortBy] = useState('newest')

    // Fetch projects from API
    useEffect(() => {
        const fetchProjects = async () => {
            try {
                const data = await apiClient.get('projects/')
                // Map API response to UI shape
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
                }))
                setProjects(mapped)
            } catch (err) {
                console.error('Failed to fetch projects:', err)
            } finally {
                setLoading(false)
            }
        }
        fetchProjects()
    }, [])

    const filtered = useMemo(() => {
        let result = [...projects]

        if (search.trim()) {
            const q = search.toLowerCase()
            result = result.filter(p =>
                p.name.toLowerCase().includes(q) ||
                p.description.toLowerCase().includes(q) ||
                p.model.toLowerCase().includes(q)
            )
        }

        if (statusFilter !== 'all') {
            result = result.filter(p => p.status === statusFilter)
        }

        if (sortBy === 'newest') {
            result.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
        } else if (sortBy === 'oldest') {
            result.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))
        } else if (sortBy === 'name') {
            result.sort((a, b) => a.name.localeCompare(b.name))
        }

        return result
    }, [projects, search, statusFilter, sortBy])

    return (
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold">Projects</h1>
                    <p className="text-sm text-muted-foreground">{projects.length} project{projects.length !== 1 ? 's' : ''} total</p>
                </div>
                <Button
                    className="gap-2 bg-primary hover:bg-primary/90 text-primary-foreground glow"
                    onClick={() => navigate('/projects/new')}
                >
                    <Plus className="w-4 h-4" />
                    New Project
                </Button>
            </div>

            {/* Toolbar */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                        placeholder="Search projects..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="pl-9 bg-background"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                        <SelectTrigger className="w-[130px] bg-background">
                            <Filter className="w-3.5 h-3.5 mr-1.5 text-muted-foreground" />
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Status</SelectItem>
                            <SelectItem value="running">Running</SelectItem>
                            <SelectItem value="completed">Complete</SelectItem>
                            <SelectItem value="failed">Error</SelectItem>
                            <SelectItem value="pending">Pending</SelectItem>
                        </SelectContent>
                    </Select>

                    <Select value={sortBy} onValueChange={setSortBy}>
                        <SelectTrigger className="w-[120px] bg-background">
                            <ArrowUpDown className="w-3.5 h-3.5 mr-1.5 text-muted-foreground" />
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="newest">Newest</SelectItem>
                            <SelectItem value="oldest">Oldest</SelectItem>
                            <SelectItem value="name">Name</SelectItem>
                        </SelectContent>
                    </Select>

                    <div className="hidden sm:flex items-center border border-border rounded-lg p-0.5 bg-background">
                        <Button
                            variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => setViewMode('grid')}
                        >
                            <LayoutGrid className="w-4 h-4" />
                        </Button>
                        <Button
                            variant={viewMode === 'table' ? 'secondary' : 'ghost'}
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => setViewMode('table')}
                        >
                            <List className="w-4 h-4" />
                        </Button>
                    </div>
                </div>
            </div>

            {/* Loading */}
            {loading ? (
                <div className="flex justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
            ) : filtered.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                        {search.trim() ? <Search className="w-5 h-5 text-muted-foreground" /> : <FolderOpen className="w-5 h-5 text-muted-foreground" />}
                    </div>
                    <p className="font-medium">{search.trim() ? 'No projects found' : 'No projects yet'}</p>
                    <p className="text-sm text-muted-foreground mt-1">
                        {search.trim() ? 'Try adjusting your search or filter' : 'Create your first project to get started'}
                    </p>
                    {!search.trim() && (
                        <Button className="mt-4 gap-2" onClick={() => navigate('/projects/new')}>
                            <Plus className="w-4 h-4" />
                            New Project
                        </Button>
                    )}
                </div>
            ) : viewMode === 'grid' ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filtered.map(project => (
                        <ProjectCard key={project.id} project={project} />
                    ))}
                </div>
            ) : (
                <div className="rounded-xl border border-border overflow-hidden">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-border bg-muted/30">
                                <th className="text-left text-xs font-medium text-muted-foreground px-4 py-3">Name</th>
                                <th className="text-left text-xs font-medium text-muted-foreground px-4 py-3 hidden md:table-cell">Model</th>
                                <th className="text-left text-xs font-medium text-muted-foreground px-4 py-3">Status</th>
                                <th className="text-left text-xs font-medium text-muted-foreground px-4 py-3 hidden md:table-cell">Mode</th>
                                <th className="text-left text-xs font-medium text-muted-foreground px-4 py-3 hidden lg:table-cell">Created</th>
                                <th className="px-4 py-3 w-10"></th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {filtered.map(project => {
                                const status = statusConfig[project.status] || statusConfig.pending
                                return (
                                    <tr
                                        key={project.id}
                                        onClick={() => navigate(`/projects/${project.id}`)}
                                        className="group cursor-pointer hover:bg-muted/30 transition-colors"
                                    >
                                        <td className="px-4 py-3">
                                            <div>
                                                <p className="text-sm font-medium group-hover:text-primary transition-colors">{project.name}</p>
                                                <p className="text-xs text-muted-foreground line-clamp-1 mt-0.5">{project.description}</p>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3 hidden md:table-cell">
                                            <span className="text-xs text-muted-foreground">{project.model}</span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <Badge variant="outline" className={`text-[11px] ${status.className}`}>
                                                {status.label}
                                            </Badge>
                                        </td>
                                        <td className="px-4 py-3 hidden md:table-cell">
                                            <span className="text-xs text-muted-foreground capitalize">{project.mode?.replace(/_/g, ' ')}</span>
                                        </td>
                                        <td className="px-4 py-3 hidden lg:table-cell">
                                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                                <Clock className="w-3 h-3" />
                                                {timeAgo(project.createdAt)}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <ArrowRight className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
