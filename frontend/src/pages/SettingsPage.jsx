import React, { useState } from 'react'
import { Save, SlidersHorizontal, Palette, CheckCircle2, Sun, Moon } from 'lucide-react'
import { useTheme } from '@/components/ThemeProvider'

export default function SettingsPage() {
    const { theme, toggleTheme } = useTheme()
    const [saved, setSaved] = useState(false)

    // Defaults
    const [defaultModel, setDefaultModel] = useState('llama-3.1-8b')
    const [defaultSamples, setDefaultSamples] = useState(5)
    const [defaultThreshold, setDefaultThreshold] = useState(70)
    const [autoRetry, setAutoRetry] = useState(true)
    const [notifications, setNotifications] = useState(true)

    const handleSave = () => {
        setSaved(true)
        setTimeout(() => setSaved(false), 2500)
    }

    return (
        <div className="min-h-screen bg-canvas-soft font-sans selection:bg-ink selection:text-canvas pb-24">
            <div className="max-w-[800px] mx-auto px-4 md:px-6 py-12 md:py-16 space-y-8">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-[32px] leading-[40px] font-semibold tracking-[-1.28px] text-ink">
                            Settings
                        </h1>
                        <p className="text-[16px] leading-[24px] text-body mt-1">
                            Configure your Modifai workspace.
                        </p>
                    </div>
                    <button
                        onClick={handleSave}
                        disabled={saved}
                        className={`
                            px-6 h-10 rounded-full text-[14px] font-medium transition-all flex items-center gap-2 shadow-sm
                            ${saved
                                ? 'bg-canvas border border-hairline text-ink'
                                : 'bg-ink text-canvas hover:bg-ink/90'}
                        `}
                    >
                        {saved ? (
                            <>
                                <CheckCircle2 className="w-4 h-4 text-[#0070f3]" />
                                Saved
                            </>
                        ) : (
                            <>
                                <Save className="w-4 h-4" />
                                Save Changes
                            </>
                        )}
                    </button>
                </div>

                {/* Pipeline Defaults */}
                <div className="bg-canvas rounded-[12px] border border-hairline shadow-[0px_1px_1px_#00000005,0px_2px_2px_#0000000a] overflow-hidden">
                    <div className="p-6 md:p-8 space-y-8">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-full border border-hairline bg-canvas-soft flex items-center justify-center shrink-0">
                                <SlidersHorizontal className="w-5 h-5 text-ink" strokeWidth={1.5} />
                            </div>
                            <div>
                                <h2 className="text-[18px] font-semibold text-ink">Pipeline Defaults</h2>
                                <p className="text-[14px] text-body">Default configuration for new projects</p>
                            </div>
                        </div>

                        <div className="space-y-8">
                            <div className="space-y-3">
                                <label className="block text-[14px] font-medium text-ink">Default Base Model</label>
                                <select
                                    value={defaultModel}
                                    onChange={(e) => setDefaultModel(e.target.value)}
                                    className="w-full h-10 px-3 bg-canvas border border-hairline rounded-[6px] text-[14px] text-ink focus:outline-none focus:ring-1 focus:ring-[#171717] focus:border-ink appearance-none"
                                    style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%23171717%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', backgroundSize: '16px' }}
                                >
                                    <option value="llama-3.1-8b">Llama 3.1 8B</option>
                                    <option value="mistral-7b">Mistral 7B</option>
                                    <option value="gemma-2-9b">Gemma 2 9B</option>
                                    <option value="phi-3-mini">Phi-3 Mini</option>
                                </select>
                            </div>

                            <div className="w-full h-[1px] bg-hairline" />

                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <label className="text-[14px] font-medium text-ink">Samples per Chunk</label>
                                    <div className="font-mono text-[12px] bg-canvas-soft border border-hairline px-2 py-0.5 rounded-[4px] text-ink">{defaultSamples}</div>
                                </div>
                                <input
                                    type="range"
                                    value={defaultSamples}
                                    onChange={(e) => setDefaultSamples(parseInt(e.target.value))}
                                    min={1} max={20} step={1}
                                    className="w-full accent-[#171717]"
                                    style={{ '--value': `${((defaultSamples - 1) / 19) * 100}%` }}
                                />
                            </div>

                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <label className="text-[14px] font-medium text-ink">Quality Threshold</label>
                                    <div className="font-mono text-[12px] bg-canvas-soft border border-hairline px-2 py-0.5 rounded-[4px] text-ink">{defaultThreshold}%</div>
                                </div>
                                <input
                                    type="range"
                                    value={defaultThreshold}
                                    onChange={(e) => setDefaultThreshold(parseInt(e.target.value))}
                                    min={0} max={100} step={5}
                                    className="w-full accent-[#171717]"
                                    style={{ '--value': `${defaultThreshold}%` }}
                                />
                            </div>

                            <div className="w-full h-[1px] bg-hairline" />

                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-[14px] font-medium text-ink">Auto-retry on failure</p>
                                    <p className="text-[12px] text-mute">Automatically retry failed pipeline steps</p>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" className="sr-only peer" checked={autoRetry} onChange={() => setAutoRetry(!autoRetry)} />
                                    <div className="w-9 h-5 bg-hairline rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white dark:after:bg-[#111111] peer-checked:after:bg-white dark:peer-checked:after:bg-[#111111] after:border-gray-300 dark:after:border-transparent after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-ink"></div>
                                </label>
                            </div>

                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-[14px] font-medium text-ink">Pipeline notifications</p>
                                    <p className="text-[12px] text-mute">Get notified when pipelines complete or fail</p>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" className="sr-only peer" checked={notifications} onChange={() => setNotifications(!notifications)} />
                                    <div className="w-9 h-5 bg-hairline rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white dark:after:bg-[#111111] peer-checked:after:bg-white dark:peer-checked:after:bg-[#111111] after:border-gray-300 dark:after:border-transparent after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-ink"></div>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Appearance */}
                <div className="bg-canvas rounded-[12px] border border-hairline shadow-[0px_1px_1px_#00000005,0px_2px_2px_#0000000a] overflow-hidden">
                    <div className="p-6 md:p-8 space-y-8">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-full border border-hairline bg-canvas-soft flex items-center justify-center shrink-0">
                                <Palette className="w-5 h-5 text-ink" strokeWidth={1.5} />
                            </div>
                            <div>
                                <h2 className="text-[18px] font-semibold text-ink">Appearance</h2>
                                <p className="text-[14px] text-body">Customize the look and feel</p>
                            </div>
                        </div>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-[14px] font-medium text-ink">Theme</p>
                                <p className="text-[12px] text-mute">Switch between dark and light mode</p>
                            </div>
                            <button
                                onClick={toggleTheme}
                                className="bg-canvas border border-hairline text-ink px-4 h-8 rounded-full text-[12px] font-medium hover:bg-canvas-soft transition-colors flex items-center gap-2"
                            >
                                {theme === 'dark' ? (
                                    <>
                                        <Moon className="w-3.5 h-3.5" />
                                        Dark
                                    </>
                                ) : (
                                    <>
                                        <Sun className="w-3.5 h-3.5" />
                                        Light
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
