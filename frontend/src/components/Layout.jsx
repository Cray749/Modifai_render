import React, { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
    LayoutDashboard,
    FolderKanban,
    Settings,
    Plus,
    ChevronLeft,
    ChevronRight,
    Hexagon,
    Menu,
    X,
    Sun,
    Moon,
} from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useTheme } from '@/components/ThemeProvider'

const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/projects', icon: FolderKanban, label: 'Projects' },
    { to: '/settings', icon: Settings, label: 'Settings' },
]

export default function Layout() {
    const [collapsed, setCollapsed] = useState(false)
    const [mobileOpen, setMobileOpen] = useState(false)
    const location = useLocation()
    const { theme, toggleTheme } = useTheme()

    return (
        <div className="flex h-screen overflow-hidden bg-canvas-soft font-sans">
            {/* Mobile overlay */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm md:hidden transition-all"
                    onClick={() => setMobileOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside
                className={`
                    fixed md:relative z-50 h-full flex flex-col
                    border-r border-hairline bg-canvas
                    transition-all duration-300 ease-in-out shadow-[1px_0px_0px_#00000005]
                    ${collapsed ? 'w-[68px]' : 'w-64'}
                    ${mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
                `}
            >
                {/* Logo */}
                <div className="flex items-center gap-3 px-6 h-16 shrink-0">
                    <div className="flex items-center justify-center shrink-0">
                        <Hexagon className="w-5 h-5 text-ink fill-ink" />
                    </div>
                    {!collapsed && (
                        <span className="text-[16px] leading-[24px] font-semibold tracking-tight text-ink">
                            Modifai
                        </span>
                    )}
                    {/* Mobile close */}
                    <button
                        className="ml-auto md:hidden w-8 h-8 flex items-center justify-center text-mute hover:text-ink"
                        onClick={() => setMobileOpen(false)}
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="w-full h-[1px] bg-hairline" />

                {/* New Project CTA */}
                <div className="px-4 py-4">
                    <NavLink to="/projects/new" onClick={() => setMobileOpen(false)}>
                        <button
                            className={`w-full flex items-center gap-2 bg-ink hover:bg-ink/90 text-canvas rounded-[6px] h-8 transition-colors ${collapsed ? 'justify-center px-0' : 'px-3'}`}
                        >
                            <Plus className="w-4 h-4 shrink-0" />
                            {!collapsed && <span className="text-[14px] font-medium">New Project</span>}
                        </button>
                    </NavLink>
                </div>

                {/* Navigation */}
                <nav className="flex-1 py-2 space-y-[2px] overflow-y-auto">
                    {navItems.map(({ to, icon: Icon, label }) => {
                        const isActive =
                            to === '/'
                                ? location.pathname === '/'
                                : location.pathname.startsWith(to)

                        const link = (
                            <NavLink
                                key={to}
                                to={to}
                                onClick={() => setMobileOpen(false)}
                                className={`
                                    flex items-center gap-3 py-[6px] transition-colors relative
                                    ${collapsed ? 'justify-center px-0 mx-4 rounded-md hover:bg-canvas-soft' : 'px-6'}
                                    ${isActive 
                                        ? (collapsed ? 'bg-canvas-soft' : 'bg-canvas-soft before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[3px] before:bg-ink') 
                                        : (collapsed ? '' : 'hover:bg-canvas-soft')
                                    }
                                `}
                            >
                                <Icon className={`w-[18px] h-[18px] shrink-0 transition-colors ${isActive ? 'text-ink' : 'text-mute'}`} strokeWidth={isActive ? 2 : 1.5} />
                                {!collapsed && (
                                    <span className={`text-[14px] leading-[20px] transition-colors ${isActive ? 'font-medium text-ink' : 'text-body'}`}>
                                        {label}
                                    </span>
                                )}
                            </NavLink>
                        )

                        if (collapsed) {
                            return (
                                <Tooltip key={to} delayDuration={0}>
                                    <TooltipTrigger asChild>{link}</TooltipTrigger>
                                    <TooltipContent side="right" className="font-medium text-[12px] bg-ink text-canvas border-none px-2 py-1">
                                        {label}
                                    </TooltipContent>
                                </Tooltip>
                            )
                        }
                        return link
                    })}
                </nav>

                {/* Bottom controls */}
                <div className="hidden md:flex items-center gap-2 px-4 py-4 border-t border-hairline">
                    <Tooltip delayDuration={0}>
                        <TooltipTrigger asChild>
                            <button
                                className={`w-8 h-8 rounded-md flex items-center justify-center text-mute hover:text-ink hover:bg-canvas-soft transition-colors ${collapsed ? 'hidden' : ''}`}
                                onClick={toggleTheme}
                            >
                                {theme === 'dark' ? (
                                    <Sun className="w-[18px] h-[18px]" strokeWidth={1.5} />
                                ) : (
                                    <Moon className="w-[18px] h-[18px]" strokeWidth={1.5} />
                                )}
                            </button>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="font-medium text-[12px] bg-ink text-canvas border-none px-2 py-1">
                            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
                        </TooltipContent>
                    </Tooltip>
                    <button
                        className={`w-8 h-8 rounded-md flex items-center justify-center text-mute hover:text-ink hover:bg-canvas-soft transition-colors ${collapsed ? 'mx-auto' : 'ml-auto'}`}
                        onClick={() => setCollapsed(!collapsed)}
                    >
                        {collapsed ? (
                            <ChevronRight className="w-[18px] h-[18px]" strokeWidth={1.5} />
                        ) : (
                            <ChevronLeft className="w-[18px] h-[18px]" strokeWidth={1.5} />
                        )}
                    </button>
                </div>
            </aside>

            {/* Main content */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {/* Top bar (mobile) */}
                <header className="flex md:hidden items-center h-14 px-4 border-b border-hairline bg-canvas shrink-0">
                    <button
                        className="w-9 h-9 flex items-center justify-center text-mute hover:text-ink"
                        onClick={() => setMobileOpen(true)}
                    >
                        <Menu className="w-5 h-5" />
                    </button>
                    <div className="flex items-center gap-2 ml-3">
                        <Hexagon className="w-4 h-4 text-ink fill-ink" />
                        <span className="text-[14px] leading-[20px] font-semibold tracking-tight text-ink">Modifai</span>
                    </div>
                </header>

                {/* Page content */}
                <main className="flex-1 overflow-y-auto">
                    <Outlet />
                </main>
            </div>
        </div>
    )
}
