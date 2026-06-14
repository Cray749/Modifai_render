import React, { createContext, useContext, useEffect, useState } from 'react'

const ThemeContext = createContext()

export function ThemeProvider({ children, defaultTheme = 'dark', storageKey = 'modifai-theme' }) {
    const [theme, setTheme] = useState(() => {
        try {
            return localStorage.getItem(storageKey) || defaultTheme
        } catch {
            return defaultTheme
        }
    })

    useEffect(() => {
        const root = document.documentElement
        root.classList.remove('light', 'dark')
        root.classList.add(theme)
        try {
            localStorage.setItem(storageKey, theme)
        } catch { }
    }, [theme, storageKey])

    const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))

    return (
        <ThemeContext.Provider value={{ theme, setTheme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useTheme() {
    const context = useContext(ThemeContext)
    if (!context) throw new Error('useTheme must be used within a ThemeProvider')
    return context
}
