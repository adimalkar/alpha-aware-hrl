import { NavLink, useLocation } from 'react-router-dom';
import { useState } from 'react';
import {
  LayoutDashboard,
  TrendingUp,
  Brain,
  Activity,
  BarChart3,
  Layers,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

const navItems = [
  { section: 'Analytics' },
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/portfolio', label: 'Portfolio', icon: TrendingUp },
  { path: '/training', label: 'Training', icon: Activity },
  { section: 'Intelligence' },
  { path: '/regimes', label: 'LLM Regimes', icon: Brain },
  { path: '/lob', label: 'LOB Heatmap', icon: Layers },
  { section: 'Research' },
  { path: '/baselines', label: 'Baselines', icon: BarChart3 },
  { path: '/config', label: 'Model Config', icon: Settings },
];

export default function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-brand">
        <div className="brand-icon">α</div>
        <div className="brand-text">
          <span className="brand-name">Alpha-Aware</span>
          <span className="brand-tag">Hierarchical RL</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item, i) => {
          if (item.section) {
            return (
              <div className="nav-section-title" key={`section-${i}`}>
                {item.section}
              </div>
            );
          }
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon />
              <span className="nav-label">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-toggle">
        <button onClick={onToggle}>
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
