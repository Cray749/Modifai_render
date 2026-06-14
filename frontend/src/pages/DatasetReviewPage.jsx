import React, { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
    ArrowLeft,
    Search,
    Download,
    Trash2,
    Pencil,
    Check,
    X,
    Loader2,
    Database,
    FileText,
    AlertCircle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { apiClient } from '@/api/client'

export default function DatasetReviewPage() {
    const { id } = useParams()
    const navigate = useNavigate()

    const [dataset, setDataset] = useState([])
    const [total, setTotal] = useState(0)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    // Search
    const [searchQuery, setSearchQuery] = useState('')
    const [searching, setSearching] = useState(false)

    // Inline editing
    const [editingId, setEditingId] = useState(null)
    const [editInstruction, setEditInstruction] = useState('')
    const [editResponse, setEditResponse] = useState('')
    const [saving, setSaving] = useState(false)

    // Deleting
    const [deletingId, setDeletingId] = useState(null)

    // ── Fetch dataset ─────────────────────────────────────────────────────────
    const fetchDataset = useCallback(async () => {
        try {
            setLoading(true)
            setError(null)
            const data = await apiClient.get(`projects/${id}/dataset`)
            setDataset(data.dataset || [])
            setTotal(data.total || 0)
        } catch (err) {
            console.error('Failed to fetch dataset:', err)
            setError('Failed to load dataset')
        } finally {
            setLoading(false)
        }
    }, [id])

    useEffect(() => {
        fetchDataset()
    }, [fetchDataset])

    // ── Search ────────────────────────────────────────────────────────────────
    const handleSearch = async () => {
        if (!searchQuery.trim()) {
            fetchDataset()
            return
        }
        try {
            setSearching(true)
            const data = await apiClient.get(`projects/${id}/dataset/search?q=${encodeURIComponent(searchQuery)}`)
            setDataset(data.results || [])
            setTotal(data.total || 0)
        } catch (err) {
            console.error('Search failed:', err)
        } finally {
            setSearching(false)
        }
    }

    // ── Edit example ──────────────────────────────────────────────────────────
    const startEdit = (index, example) => {
        setEditingId(index)
        setEditInstruction(example.instruction || '')
        setEditResponse(example.response || '')
    }

    const cancelEdit = () => {
        setEditingId(null)
        setEditInstruction('')
        setEditResponse('')
    }

    const saveEdit = async (index) => {
        try {
            setSaving(true)
            await apiClient.put(`projects/${id}/dataset/${index}`, {
                instruction: editInstruction,
                response: editResponse,
            })
            // Update locally
            setDataset(prev => prev.map((item, i) =>
                i === index ? { ...item, instruction: editInstruction, response: editResponse } : item
            ))
            setEditingId(null)
        } catch (err) {
            console.error('Failed to save:', err)
            alert('Failed to save changes')
        } finally {
            setSaving(false)
        }
    }

    // ── Delete example ────────────────────────────────────────────────────────
    const handleDelete = async (index) => {
        if (!confirm('Delete this training example?')) return
        try {
            setDeletingId(index)
            await apiClient.delete(`projects/${id}/dataset/${index}`)
            setDataset(prev => prev.filter((_, i) => i !== index))
            setTotal(prev => prev - 1)
        } catch (err) {
            console.error('Failed to delete:', err)
            alert('Failed to delete example')
        } finally {
            setDeletingId(null)
        }
    }

    // ── Export ─────────────────────────────────────────────────────────────────
    const handleExport = async () => {
        try {
            const data = await apiClient.get(`projects/${id}/dataset/export`)
            if (data.download_url) {
                window.open(data.download_url, '_blank')
            }
        } catch (err) {
            console.error('Export failed:', err)
            alert('Failed to generate download link')
        }
    }

    return (
        <div className="p-6 md:p-8 max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="flex items-center gap-3">
                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => navigate(`/projects/${id}`)}>
                        <ArrowLeft className="w-4 h-4" />
                    </Button>
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <Database className="w-6 h-6 text-primary" />
                            Dataset Review
                        </h1>
                        <p className="text-sm text-muted-foreground">{total} training example{total !== 1 ? 's' : ''}</p>
                    </div>
                </div>

                <Button variant="outline" className="gap-2" onClick={handleExport}>
                    <Download className="w-4 h-4" />
                    Export JSONL
                </Button>
            </div>

            {/* Search Bar */}
            <div className="flex items-center gap-2">
                <div className="relative flex-1 max-w-lg">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                        placeholder="Search examples..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        className="pl-9 bg-background"
                    />
                </div>
                <Button variant="outline" onClick={handleSearch} disabled={searching}>
                    {searching ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Search'}
                </Button>
                {searchQuery && (
                    <Button variant="ghost" size="sm" onClick={() => { setSearchQuery(''); fetchDataset() }}>
                        Clear
                    </Button>
                )}
            </div>

            {/* Content */}
            {loading ? (
                <div className="flex justify-center py-20">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <AlertCircle className="w-8 h-8 text-red-500 mb-4" />
                    <p className="font-medium">{error}</p>
                    <Button variant="outline" className="mt-4" onClick={fetchDataset}>Retry</Button>
                </div>
            ) : dataset.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                        <FileText className="w-5 h-5 text-muted-foreground" />
                    </div>
                    <p className="font-medium">No examples found</p>
                    <p className="text-sm text-muted-foreground mt-1">
                        {searchQuery ? 'Try a different search query' : 'The dataset is empty or hasn\'t been generated yet'}
                    </p>
                </div>
            ) : (
                <div className="space-y-3">
                    {dataset.map((example, index) => (
                        <Card key={index} className="border-border bg-card group">
                            <CardContent className="p-4">
                                {editingId === index ? (
                                    /* ── Edit Mode ────────────────── */
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Instruction</label>
                                            <Textarea
                                                value={editInstruction}
                                                onChange={(e) => setEditInstruction(e.target.value)}
                                                rows={3}
                                                className="bg-background resize-none text-sm"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Response</label>
                                            <Textarea
                                                value={editResponse}
                                                onChange={(e) => setEditResponse(e.target.value)}
                                                rows={4}
                                                className="bg-background resize-none text-sm"
                                            />
                                        </div>
                                        <div className="flex items-center gap-2 justify-end">
                                            <Button variant="ghost" size="sm" onClick={cancelEdit} className="gap-1.5">
                                                <X className="w-3.5 h-3.5" />
                                                Cancel
                                            </Button>
                                            <Button size="sm" onClick={() => saveEdit(index)} disabled={saving} className="gap-1.5">
                                                {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
                                                Save
                                            </Button>
                                        </div>
                                    </div>
                                ) : (
                                    /* ── View Mode ────────────────── */
                                    <div className="space-y-3">
                                        <div className="flex items-start justify-between gap-4">
                                            <div className="space-y-2 flex-1 min-w-0">
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-1">Instruction</p>
                                                    <p className="text-sm leading-relaxed">{example.instruction}</p>
                                                </div>
                                                <Separator />
                                                <div>
                                                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-1">Response</p>
                                                    <p className="text-sm leading-relaxed text-muted-foreground">{example.response}</p>
                                                </div>
                                            </div>

                                            {/* Actions */}
                                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-7 w-7"
                                                    onClick={() => startEdit(index, example)}
                                                >
                                                    <Pencil className="w-3.5 h-3.5" />
                                                </Button>
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-7 w-7 text-destructive hover:text-destructive"
                                                    onClick={() => handleDelete(index)}
                                                    disabled={deletingId === index}
                                                >
                                                    {deletingId === index
                                                        ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                        : <Trash2 className="w-3.5 h-3.5" />
                                                    }
                                                </Button>
                                            </div>
                                        </div>

                                        {/* Metadata row */}
                                        <div className="flex items-center gap-2 pt-1">
                                            <Badge variant="outline" className="text-[10px] font-mono">#{index + 1}</Badge>
                                            {example.confidence != null && (
                                                <Badge
                                                    variant="outline"
                                                    className={`text-[10px] ${example.confidence > 0.7
                                                        ? 'border-emerald-500/30 text-emerald-500'
                                                        : example.confidence > 0.4
                                                            ? 'border-amber-500/30 text-amber-500'
                                                            : 'border-red-500/30 text-red-500'
                                                        }`}
                                                >
                                                    {Math.round(example.confidence * 100)}% confidence
                                                </Badge>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    )
}
