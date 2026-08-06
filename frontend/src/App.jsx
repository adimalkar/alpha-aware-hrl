import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import DashboardPage from './pages/DashboardPage';
import PortfolioPage from './pages/PortfolioPage';
import TrainingPage from './pages/TrainingPage';
import RegimesPage from './pages/RegimesPage';
import LOBPage from './pages/LOBPage';
import BaselinesPage from './pages/BaselinesPage';
import ConfigPage from './pages/ConfigPage';

export default function App() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="app-layout">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
      <main className={`main-content ${collapsed ? 'collapsed' : ''}`}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/regimes" element={<RegimesPage />} />
          <Route path="/lob" element={<LOBPage />} />
          <Route path="/baselines" element={<BaselinesPage />} />
          <Route path="/config" element={<ConfigPage />} />
        </Routes>
      </main>
    </div>
  );
}
