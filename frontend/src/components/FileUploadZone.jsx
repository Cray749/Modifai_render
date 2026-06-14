import React, { useState, useRef, useCallback } from 'react'
import { Upload, X, FileText, Image, File } from 'lucide-react'

const ACCEPTED_TYPES = {
    'application/pdf': 'PDF',
    'image/png': 'PNG',
    'image/jpeg': 'JPG',
    'image/jpg': 'JPG',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
    'text/plain': 'TXT',
}

const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB

function getFileIcon(type) {
    if (type.startsWith('image/')) return Image
    if (type === 'application/pdf') return FileText
    return File
}

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileUploadZone({ files, onFilesChange }) {
    const [dragActive, setDragActive] = useState(false)
    const inputRef = useRef(null)

    const handleFiles = useCallback((newFiles) => {
        const validFiles = Array.from(newFiles).filter(file => {
            if (!Object.keys(ACCEPTED_TYPES).includes(file.type)) return false
            if (file.size > MAX_FILE_SIZE) return false
            return true
        }).map(file => ({
            id: `${file.name}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            file,
            name: file.name,
            size: file.size,
            type: file.type,
            progress: 100, // simulated
        }))

        onFilesChange([...files, ...validFiles])
    }, [files, onFilesChange])

    const handleDrag = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files?.length) {
            handleFiles(e.dataTransfer.files)
        }
    }, [handleFiles])

    const removeFile = (id) => {
        onFilesChange(files.filter(f => f.id !== id))
    }

    return (
        <div className="space-y-4">
            {/* Drop zone */}
            <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`
                    relative flex flex-col items-center justify-center
                    rounded-[8px] border p-12 cursor-pointer
                    transition-all duration-200
                    ${dragActive
                        ? 'border-solid border-ink bg-canvas scale-[1.01]'
                        : 'border-dashed border-hairline bg-canvas-soft hover:border-hairline-strong'
                    }
                `}
            >
                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept={Object.keys(ACCEPTED_TYPES).join(',')}
                    className="hidden"
                    onChange={(e) => {
                        if (e.target.files?.length) handleFiles(e.target.files)
                        e.target.value = ''
                    }}
                />

                <div className={`w-12 h-12 rounded-full border flex items-center justify-center mb-4 transition-colors duration-200 ${dragActive ? 'bg-ink border-ink' : 'bg-canvas border-hairline'
                    }`}>
                    <Upload className={`w-5 h-5 transition-colors duration-200 ${dragActive ? 'text-canvas' : 'text-ink'
                        }`} strokeWidth={1.5} />
                </div>

                <p className="text-[14px] leading-[20px] font-medium text-ink mb-1">
                    {dragActive ? 'Drop files here' : 'Drag & drop files here'}
                </p>
                <p className="font-mono text-[12px] text-mute uppercase tracking-wider mt-2">
                    PDF, PNG, JPG, DOCX, TXT · Max 50MB
                </p>
            </div>

            {/* File list */}
            {files.length > 0 && (
                <div className="space-y-3 mt-6">
                    <div className="flex items-center justify-between">
                        <p className="font-mono text-[12px] text-mute uppercase tracking-wider">
                            {files.length} file{files.length !== 1 ? 's' : ''} selected
                        </p>
                        <button
                            className="text-[12px] text-body hover:text-ink transition-colors"
                            onClick={(e) => { e.stopPropagation(); onFilesChange([]) }}
                        >
                            Clear all
                        </button>
                    </div>

                    <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                        {files.map((f) => {
                            const Icon = getFileIcon(f.type)
                            return (
                                <div
                                    key={f.id}
                                    className="flex items-center gap-4 rounded-[6px] border border-hairline bg-canvas px-4 py-3 group hover:border-hairline-strong transition-colors"
                                >
                                    <div className="w-10 h-10 rounded-full border border-hairline bg-canvas-soft flex items-center justify-center shrink-0">
                                        <Icon className="w-4 h-4 text-ink" strokeWidth={1.5} />
                                    </div>
                                    <div className="flex-1 min-w-0 flex flex-col gap-0.5">
                                        <p className="text-[14px] leading-[20px] font-medium text-ink truncate">{f.name}</p>
                                        <p className="font-mono text-[10px] text-mute uppercase tracking-wider">
                                            {formatSize(f.size)} · {ACCEPTED_TYPES[f.type] || 'FILE'}
                                        </p>
                                    </div>
                                    <button
                                        className="h-8 w-8 shrink-0 flex items-center justify-center rounded-md opacity-0 group-hover:opacity-100 transition-opacity text-mute hover:text-ink hover:bg-canvas-soft"
                                        onClick={(e) => { e.stopPropagation(); removeFile(f.id) }}
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}
