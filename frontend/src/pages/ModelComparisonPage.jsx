import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
    ArrowLeft,
    Send,
    Copy,
    Check,
    Zap,
    Clock,
    Brain,
    Sparkles,
    ChevronDown,
    ChevronUp,
    BarChart3,
    Loader2,
    MessageSquare,
    Bot,
    Cpu,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { apiClient } from '@/api/client'

// ── Typing animation component ───────────────────────────────────────────────
function TypewriterText({ text, speed = 12 }) {
    const [displayed, setDisplayed] = useState('')
    const [done, setDone] = useState(false)

    useEffect(() => {
        setDisplayed('')
        setDone(false)
        if (!text) return
        let i = 0
        const interval = setInterval(() => {
            i++
            setDisplayed(text.slice(0, i))
            if (i >= text.length) {
                clearInterval(interval)
                setDone(true)
            }
        }, speed)
        return () => clearInterval(interval)
    }, [text, speed])

    return (
        <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {displayed}
            {!done && <span className="inline-block w-0.5 h-4 bg-primary animate-pulse ml-0.5" />}
        </div>
    )
}

// ── Copy button ──────────────────────────────────────────────────────────────
function CopyButton({ text }) {
    const [copied, setCopied] = useState(false)

    const handleCopy = async () => {
        await navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    return (
        <button
            onClick={handleCopy}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
            title="Copy to clipboard"
        >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
    )
}

// ── Latency comparison bar ───────────────────────────────────────────────────
function LatencyBar({ baseMs, ftMs }) {
    const maxMs = Math.max(baseMs, ftMs, 1)
    return (
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <span>Base Model</span>
                    <span className="font-mono">{baseMs}ms</span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                        className="h-full rounded-full bg-blue-500 transition-all duration-700"
                        style={{ width: `${(baseMs / maxMs) * 100}%` }}
                    />
                </div>
            </div>
            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <span>Fine-Tuned</span>
                    <span className="font-mono">{ftMs}ms</span>
                </div>
                <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                        className="h-full rounded-full bg-emerald-500 transition-all duration-700"
                        style={{ width: `${(ftMs / maxMs) * 100}%` }}
                    />
                </div>
            </div>
        </div>
    )
}

// ── Response Panel ───────────────────────────────────────────────────────────
function ResponsePanel({ title, icon: Icon, result, loading, accentColor, animate }) {
    const hasError = result?.error && !result?.response

    return (
        <Card className={`flex-1 border-2 transition-all duration-300 ${loading ? `border-${accentColor}/30 shadow-lg shadow-${accentColor}/5` : 'border-border'}`}>
            <div className={`flex items-center justify-between px-5 py-3 border-b border-border bg-gradient-to-r from-${accentColor}/5 to-transparent`}>
                <div className="flex items-center gap-2">
                    <div className={`w-7 h-7 rounded-lg bg-${accentColor}/15 flex items-center justify-center`}>
                        <Icon className={`w-4 h-4 text-${accentColor}`} />
                    </div>
                    <span className="font-semibold text-sm">{title}</span>
                </div>
                <div className="flex items-center gap-2">
                    {result && !loading && (
                        <>
                            <Badge variant="outline" className="text-[10px] font-mono gap-1 px-2 py-0.5">
                                <Clock className="w-3 h-3" />
                                {result.latency_ms}ms
                            </Badge>
                            <CopyButton text={result.response} />
                        </>
                    )}
                </div>
            </div>
            <CardContent className="p-5 min-h-[200px]">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-[200px] gap-3">
                        <div className="relative">
                            <div className={`absolute inset-0 rounded-full bg-${accentColor}/20 animate-ping`} />
                            <Loader2 className={`w-8 h-8 animate-spin text-${accentColor}`} />
                        </div>
                        <span className="text-xs text-muted-foreground animate-pulse">Generating response...</span>
                    </div>
                ) : result ? (
                    hasError ? (
                        <div className="text-sm text-muted-foreground italic">{result.response || result.error}</div>
                    ) : animate ? (
                        <TypewriterText text={result.response} speed={8} />
                    ) : (
                        <div className="whitespace-pre-wrap text-sm leading-relaxed">{result.response}</div>
                    )
                ) : (
                    <div className="flex flex-col items-center justify-center h-[200px] text-muted-foreground/50 gap-2">
                        <MessageSquare className="w-8 h-8" />
                        <span className="text-xs">Send a prompt to see the response</span>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function ModelComparisonPage() {
    const { id } = useParams()
    const navigate = useNavigate()
    const textareaRef = useRef(null)

    const [prompt, setPrompt] = useState('')
    const [systemPrompt, setSystemPrompt] = useState('')
    const [showSystemPrompt, setShowSystemPrompt] = useState(false)
    const [loading, setLoading] = useState(false)
    const [history, setHistory] = useState([])
    const [currentResult, setCurrentResult] = useState(null)
    const [animating, setAnimating] = useState(false)
    const [projectName, setProjectName] = useState('')

    // Fetch project name
    useEffect(() => {
        const fetchProject = async () => {
            try {
                const data = await apiClient.get(`projects/${id}`)
                setProjectName(data.name)
            } catch { /* ignore */ }
        }
        fetchProject()
    }, [id])

    const handleCompare = async () => {
        if (!prompt.trim() || loading) return

        setLoading(true)
        setCurrentResult(null)
        setAnimating(true)

        try {
            const data = await apiClient.post('compare/', {
                project_id: id,
                prompt: prompt.trim(),
                system_prompt: systemPrompt.trim() || null,
            }, { timeout: 120_000 })

            setCurrentResult(data)

            // Push to history after animation settles
            setTimeout(() => {
                setAnimating(false)
            }, Math.max(data.base_model.response.length, data.fine_tuned.response.length) * 8 + 500)
        } catch (err) {
            console.error('Compare error:', err)
            setCurrentResult({
                base_model: { response: 'Error occurred', latency_ms: 0, model_id: 'error', error: String(err) },
                fine_tuned: { response: 'Error occurred', latency_ms: 0, model_id: 'error', error: String(err) },
            })
            setAnimating(false)
        } finally {
            setLoading(false)
        }
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
            e.preventDefault()
            handleCompare()
        }
    }

    const pushToHistory = () => {
        if (currentResult) {
            setHistory(prev => [{ prompt, systemPrompt, result: currentResult, timestamp: Date.now() }, ...prev])
        }
    }

    // When user sends a new prompt, push the old result to history
    const handleNewPrompt = async () => {
        pushToHistory()
        handleCompare()
    }

    return (
        <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
            {/* ── Header ── */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Button variant="ghost" size="icon" onClick={() => navigate(`/projects/${id}`)}>
                        <ArrowLeft className="w-4 h-4" />
                    </Button>
                    <div>
                        <div className="flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-primary" />
                            <h1 className="text-xl font-bold">Model Comparison</h1>
                        </div>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            {projectName || 'Loading...'} — Base vs Fine-Tuned side-by-side
                        </p>
                    </div>
                </div>
                <Badge variant="outline" className="gap-1.5 px-3 py-1.5 text-xs">
                    <BarChart3 className="w-3.5 h-3.5" />
                    {history.length} comparison{history.length !== 1 ? 's' : ''}
                </Badge>
            </div>

            {/* ── Prompt Input ── */}
            <Card className="border-2 border-primary/20 bg-gradient-to-br from-primary/5 via-transparent to-transparent">
                <CardContent className="p-5 space-y-3">
                    {/* System prompt toggle */}
                    <button
                        onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                        {showSystemPrompt ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        System Prompt (optional)
                    </button>

                    {showSystemPrompt && (
                        <textarea
                            value={systemPrompt}
                            onChange={e => setSystemPrompt(e.target.value)}
                            placeholder="You are a helpful assistant specialized in..."
                            className="w-full bg-muted/50 border border-border rounded-lg px-4 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/30 placeholder:text-muted-foreground/50"
                            rows={2}
                        />
                    )}

                    {/* Main prompt */}
                    <div className="relative">
                        <textarea
                            ref={textareaRef}
                            value={prompt}
                            onChange={e => setPrompt(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Enter your prompt here... (⌘+Enter to send)"
                            className="w-full bg-background border border-border rounded-xl px-5 py-4 pr-24 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/40 placeholder:text-muted-foreground/50 min-h-[80px]"
                            rows={3}
                        />
                        <Button
                            size="sm"
                            disabled={!prompt.trim() || loading}
                            onClick={currentResult ? handleNewPrompt : handleCompare}
                            className="absolute right-3 bottom-3 gap-1.5 bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/20"
                        >
                            {loading ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                                <Send className="w-3.5 h-3.5" />
                            )}
                            Compare
                        </Button>
                    </div>
                </CardContent>
            </Card>

            {/* ── Side-by-Side Response Panels ── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <ResponsePanel
                    title="Base Model"
                    icon={Bot}
                    result={currentResult?.base_model}
                    loading={loading}
                    accentColor="blue-500"
                    animate={animating}
                />
                <ResponsePanel
                    title="Fine-Tuned Model"
                    icon={Cpu}
                    result={currentResult?.fine_tuned}
                    loading={loading}
                    accentColor="emerald-500"
                    animate={animating}
                />
            </div>

            {/* ── Latency Comparison ── */}
            {currentResult && !loading && (
                <Card>
                    <CardContent className="p-4">
                        <div className="flex items-center gap-2 mb-3">
                            <Zap className="w-4 h-4 text-amber-400" />
                            <span className="text-sm font-semibold">Performance Comparison</span>
                        </div>
                        <LatencyBar
                            baseMs={currentResult.base_model.latency_ms}
                            ftMs={currentResult.fine_tuned.latency_ms}
                        />
                        <div className="grid grid-cols-2 gap-4 mt-4 text-xs text-muted-foreground">
                            <div className="flex items-center justify-between bg-muted/30 rounded-lg p-2.5">
                                <span>Response length</span>
                                <span className="font-mono">{currentResult.base_model.response.length} chars</span>
                            </div>
                            <div className="flex items-center justify-between bg-muted/30 rounded-lg p-2.5">
                                <span>Response length</span>
                                <span className="font-mono">{currentResult.fine_tuned.response.length} chars</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* ── Comparison History ── */}
            {history.length > 0 && (
                <div className="space-y-4">
                    <div className="flex items-center gap-2">
                        <Brain className="w-4 h-4 text-muted-foreground" />
                        <h2 className="text-sm font-semibold text-muted-foreground">Previous Comparisons</h2>
                    </div>
                    {history.map((item, idx) => (
                        <Card key={idx} className="border-border/60 opacity-75 hover:opacity-100 transition-opacity">
                            <CardContent className="p-4 space-y-3">
                                <div className="flex items-center justify-between">
                                    <div className="text-xs text-muted-foreground font-mono">
                                        {new Date(item.timestamp).toLocaleTimeString()}
                                    </div>
                                    <LatencyBar
                                        baseMs={item.result.base_model.latency_ms}
                                        ftMs={item.result.fine_tuned.latency_ms}
                                    />
                                </div>
                                <div className="bg-muted/30 rounded-lg p-3 text-sm">
                                    <span className="text-muted-foreground text-xs font-medium">Prompt:</span>
                                    <p className="mt-1 line-clamp-2">{item.prompt}</p>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-blue-500/5 border border-blue-500/10 rounded-lg p-3">
                                        <span className="text-[10px] text-blue-400 font-medium">BASE MODEL</span>
                                        <p className="text-xs mt-1 line-clamp-4 text-muted-foreground">{item.result.base_model.response}</p>
                                    </div>
                                    <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-3">
                                        <span className="text-[10px] text-emerald-400 font-medium">FINE-TUNED</span>
                                        <p className="text-xs mt-1 line-clamp-4 text-muted-foreground">{item.result.fine_tuned.response}</p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    )
}
