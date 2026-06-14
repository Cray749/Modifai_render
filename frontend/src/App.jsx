import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeProvider } from '@/components/ThemeProvider'
import Layout from '@/components/Layout'
import DashboardPage from '@/pages/DashboardPage'
import ProjectsListPage from '@/pages/ProjectsListPage'
import NewProjectPage from '@/pages/NewProjectPage'
import ProjectDetailPage from '@/pages/ProjectDetailPage'
import DatasetReviewPage from '@/pages/DatasetReviewPage'
import ModelComparisonPage from '@/pages/ModelComparisonPage'
import SettingsPage from '@/pages/SettingsPage'

function App() {
    return (
        <ThemeProvider defaultTheme="dark">
            <TooltipProvider>
                <BrowserRouter>
                    <Routes>
                        <Route element={<Layout />}>
                            <Route path="/" element={<DashboardPage />} />
                            <Route path="/projects" element={<ProjectsListPage />} />
                            <Route path="/projects/new" element={<NewProjectPage />} />
                            <Route path="/projects/:id" element={<ProjectDetailPage />} />
                            <Route path="/projects/:id/dataset" element={<DatasetReviewPage />} />
                            <Route path="/projects/:id/compare" element={<ModelComparisonPage />} />
                            <Route path="/settings" element={<SettingsPage />} />
                        </Route>
                    </Routes>
                </BrowserRouter>
            </TooltipProvider>
        </ThemeProvider>
    )
}

export default App