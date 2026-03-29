import { useState, createContext, useContext, useRef, useEffect } from 'react';
import { Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import {
  User,
  Scan,
  Calendar,
  FileText,
  Search,
  Bell,
  Moon,
  Sun,
  ChevronRight,
  Activity,
  Eye,
  Brain,
  Upload,
  Filter,
  ChevronLeft,
  ChevronRight as ChevronRightIcon,
  Plus,
  Camera,
  Shield,
  HelpCircle,
  Send,
  MessageSquare,
  Users,
  ClipboardList,
  History,
  Lock,
  Pill,
  FolderOpen,
  Clock,
  Stethoscope,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

// Dark Mode Context
const DarkModeContext = createContext({
  isDark: false,
  toggleDark: () => {},
});

export default function MainApp() {
  const [isDark, setIsDark] = useState(false);

  const toggleDark = () => setIsDark(!isDark);

  return (
    <DarkModeContext.Provider value={{ isDark, toggleDark }}>
      <div className={`min-h-screen transition-colors duration-300 ${isDark ? 'dark bg-navy' : 'bg-offwhite'}`}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/health" element={<HealthPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/patients" element={<PatientsPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/schedule" element={<SchedulePage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </div>
    </DarkModeContext.Provider>
  );
}

// Top Navigation Component
function TopNav() {
  const { isDark, toggleDark } = useContext(DarkModeContext);
  const navigate = useNavigate();

  const navItems = [
    { label: 'Dashboard', path: '/app' },
    { label: 'Scan', path: '/app/health' },
    { label: 'AI Chat', path: '/app/chat' },
    { label: 'Patients', path: '/app/patients' },
    { label: 'Schedule', path: '/app/schedule' },
    { label: 'Reports', path: '/app/reports' },
  ];

  return (
    <header className={`sticky top-0 z-50 border-b ${isDark ? 'bg-navy/90 border-white/10' : 'bg-white/90 border-navy/5'} backdrop-blur-md`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-mint/15 flex items-center justify-center">
              <Scan className="w-5 h-5 text-mint" />
            </div>
            <span className={`font-semibold text-lg ${isDark ? 'text-white' : 'text-navy'}`}>
              OptiScan
            </span>
          </div>

          {/* Center Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/app'}
                className={({ isActive }) =>
                  `px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-mint/15 text-mint'
                      : isDark
                      ? 'text-white/70 hover:text-white hover:bg-white/5'
                      : 'text-navy/70 hover:text-navy hover:bg-navy/5'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          {/* Right Actions */}
          <div className="flex items-center gap-3">
            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDark}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                isDark ? 'bg-white/10 text-white hover:bg-white/20' : 'bg-navy/5 text-navy hover:bg-navy/10'
              }`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>

            {/* Notifications */}
            <button className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors relative ${
              isDark ? 'bg-white/10 text-white hover:bg-white/20' : 'bg-navy/5 text-navy hover:bg-navy/10'
            }`}>
              <Bell className="w-5 h-5" />
              <span className="absolute top-2 right-2 w-2 h-2 bg-mint rounded-full" />
            </button>

            {/* Profile - Clickable */}
            <button 
              onClick={() => navigate('/app/profile')}
              className="flex items-center gap-2 pl-2"
            >
              <Avatar className="w-9 h-9 ring-2 ring-mint/30">
                <AvatarImage src="/avatar_02.jpg" />
                <AvatarFallback className="bg-mint/15 text-mint text-sm">DR</AvatarFallback>
              </Avatar>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

// Dashboard Page - Main Entry with Disease Classification Cards
function DashboardPage() {
  const { isDark } = useContext(DarkModeContext);
  const navigate = useNavigate();

  // The 4 eye diseases from the backend diagram
  const diseases = [
    {
      id: 'dr',
      name: 'Diabetic Retinopathy',
      description: 'AI-powered detection of diabetic eye disease',
      icon: Eye,
      color: 'from-blue-500/20 to-blue-600/10',
      iconColor: 'text-blue-500',
    },
    {
      id: 'glaucoma',
      name: 'Glaucoma',
      description: 'Early detection of optic nerve damage',
      icon: Brain,
      color: 'from-purple-500/20 to-purple-600/10',
      iconColor: 'text-purple-500',
    },
    {
      id: 'myopia',
      name: 'Pathological Myopia',
      description: 'Identify degenerative myopia conditions',
      icon: Activity,
      color: 'from-amber-500/20 to-amber-600/10',
      iconColor: 'text-amber-500',
    },

  ];

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-10">
          <h1 className={`text-3xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
            Eye Disease Detection
          </h1>
          <p className={`text-lg ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
            Select a disease type to start AI-powered retinal analysis
          </p>
        </div>

        {/* Search Bar - Centered */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className={`relative rounded-2xl ${isDark ? 'bg-white/10' : 'bg-white'} card-shadow`}>
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${isDark ? 'text-white/40' : 'text-navy/40'}`} />
            <input
              type="text"
              placeholder="Search patients, scans, or reports..."
              className={`w-full pl-12 pr-4 py-4 rounded-2xl text-base focus:outline-none ${
                isDark 
                  ? 'bg-transparent text-white placeholder:text-white/40' 
                  : 'bg-transparent text-navy placeholder:text-navy/40'
              }`}
            />
          </div>
        </div>

        {/* Disease Classification Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {diseases.map((disease) => (
            <button
              key={disease.id}
              onClick={() => navigate('/app/health')}
              className={`group relative overflow-hidden rounded-3xl p-8 text-left transition-all duration-300 hover:scale-[1.02] hover:shadow-xl ${
                isDark ? 'bg-white/5' : 'bg-white'
              } card-shadow`}
            >
              {/* Gradient Background */}
              <div className={`absolute inset-0 bg-gradient-to-br ${disease.color} opacity-50`} />
              
              <div className="relative z-10">
                <div className={`w-14 h-14 rounded-2xl ${isDark ? 'bg-white/10' : 'bg-white'} flex items-center justify-center mb-6 shadow-sm`}>
                  <disease.icon className={`w-7 h-7 ${disease.iconColor}`} />
                </div>
                
                <h3 className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
                  {disease.name}
                </h3>
                <p className={`text-sm mb-4 ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                  {disease.description}
                </p>
                
                <div className="flex items-center gap-2 text-mint font-medium text-sm">
                  Start Analysis
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Quick Actions */}
        <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
          <h2 className={`text-xl font-bold mb-6 ${isDark ? 'text-white' : 'text-navy'}`}>
            Quick Actions
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { icon: Upload, label: 'New Scan', action: () => navigate('/app/health') },
              { icon: Calendar, label: 'Schedule', action: () => navigate('/app/schedule') },
              { icon: FileText, label: 'Reports', action: () => navigate('/app/reports') },
              { icon: User, label: 'Profile', action: () => navigate('/app/profile') },
            ].map((item, i) => (
              <button
                key={i}
                onClick={item.action}
                className={`flex flex-col items-center gap-3 p-6 rounded-2xl transition-colors ${
                  isDark 
                    ? 'bg-white/5 hover:bg-white/10 text-white' 
                    : 'bg-navy/5 hover:bg-navy/10 text-navy'
                }`}
              >
                <div className="w-12 h-12 rounded-xl bg-mint/15 flex items-center justify-center">
                  <item.icon className="w-6 h-6 text-mint" />
                </div>
                <span className="text-sm font-medium">{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface DiseaseResult {
  probability: number;
  severity: string;
  description: string;
}

interface PredictResponse {
  scan_id: string;
  inference_ms: number;
  anomaly_score: number;
  predictions: {
    diabetic_retinopathy: DiseaseResult;
    glaucoma: DiseaseResult;
    pathologic_myopia: DiseaseResult;
  };
  meta: {
    primary_diagnosis: string;
    primary_probability: number;
    risk_level: 'Low' | 'Moderate' | 'High';
  };
  attention_map_b64: string; // base64 PNG
}

const BACKEND_URL = 'http://localhost:8000';

// ─── Sub-components ───────────────────────────────────────────────────────────

function ProbabilityBar({
  label,
  probability,
  severity,
  description,
  color,
  isDark,
}: {
  label: string;
  probability: number;
  severity: string;
  description: string;
  color: string;
  isDark: boolean;
}) {
  const pct = Math.round(probability * 100);
  return (
    <div className={`p-4 rounded-2xl ${isDark ? 'bg-white/5' : 'bg-navy/5'}`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>{label}</span>
        <span className={`text-sm font-bold ${color}`}>{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-navy/10 overflow-hidden mb-2">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color.replace('text-', 'bg-')}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className={`text-xs font-medium mb-0.5 ${color}`}>{severity}</p>
      <p className={`text-xs leading-snug ${isDark ? 'text-white/50' : 'text-navy/50'}`}>{description}</p>
    </div>
  );
}

// ─── Health Page ──────────────────────────────────────────────────────────────

function HealthPage() {
  const { isDark } = useContext(DarkModeContext);
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<'fundus' | 'anomaly' | null>(null);

  // ── File selection ─────────────────────────────────────────────────────────
  const handleFile = (file: File) => {
    if (!['image/jpeg', 'image/jpg', 'image/png'].includes(file.type)) {
      setError('Only JPEG and PNG images are supported.');
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  // ── Submit to backend ──────────────────────────────────────────────────────
  const runAnalysis = async () => {
    if (!selectedFile) return;
    setIsAnalysing(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', selectedFile);

      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail ?? 'Server error');
      }

      const data: PredictResponse = await res.json();
      setResult(data);

      // Save scan context for AI Chat
      const ctx = [
        `Risk Level: ${data.meta.risk_level}`,
        `Primary Diagnosis: ${data.meta.primary_diagnosis} (${Math.round(data.meta.primary_probability * 100)}%)`,
        `Anomaly Score: ${Math.round(data.anomaly_score * 100)}%`,
        ...Object.entries(data.predictions).map(([k, v]) => {
          const d = v as DiseaseResult;
          return `${k}: ${Math.round(d.probability * 100)}% - ${d.severity} - ${d.description}`;
        }),
      ].join('\n');
      sessionStorage.setItem('retinai_scan_context', ctx);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error. Is the backend running?');
    } finally {
      setIsAnalysing(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Risk badge colour ──────────────────────────────────────────────────────
  const riskColour =
    result?.meta.risk_level === 'High'
      ? 'bg-red-500/15 text-red-500'
      : result?.meta.risk_level === 'Moderate'
      ? 'bg-amber-500/15 text-amber-500'
      : 'bg-mint/15 text-mint';

  const diseaseLabels: Record<string, { label: string; color: string }> = {
    diabetic_retinopathy: { label: 'Diabetic Retinopathy', color: 'text-blue-500' },
    glaucoma:              { label: 'Glaucoma',              color: 'text-purple-500' },
    pathologic_myopia:     { label: 'Pathologic Myopia',     color: 'text-amber-500' },
  };

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className={`text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
            Retinal Scan Analysis
          </h1>
          <p className={`${isDark ? 'text-white/60' : 'text-navy/60'}`}>
            Upload a retinal fundus image for AI-powered multi-disease detection
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* ── Left: Upload ── */}
          <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow flex flex-col gap-6`}>
            <h2 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>
              Upload Fundus Image
            </h2>

            {/* Hidden real file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              className="hidden"
              onChange={onFileChange}
            />

            {/* Drop zone / preview */}
            {!selectedFile ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${
                  isDragging
                    ? 'border-mint bg-mint/10 scale-[1.01]'
                    : isDark
                    ? 'border-white/20 hover:border-mint/40 hover:bg-white/5'
                    : 'border-navy/20 hover:border-mint/40 hover:bg-mint/5'
                }`}
              >
                <div className="w-16 h-16 rounded-2xl bg-mint/15 flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-8 h-8 text-mint" />
                </div>
                <p className={`font-medium mb-1 ${isDark ? 'text-white' : 'text-navy'}`}>
                  {isDragging ? 'Drop it here' : 'Click to upload or drag & drop'}
                </p>
                <p className={`text-sm ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                  JPEG or PNG — max 50 MB
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Image thumbnail */}
                <div className="relative rounded-2xl overflow-hidden h-56 cursor-pointer" onClick={() => setLightbox('fundus')}>
                  <img
                    src={previewUrl!}
                    alt="Selected fundus image"
                    className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-3 left-3 px-3 py-1 bg-black/50 rounded-full text-xs text-white backdrop-blur-sm">
                    {selectedFile.name}
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/30">
                    <span className="text-white text-sm font-medium bg-black/50 px-4 py-2 rounded-full backdrop-blur-sm">Click to expand</span>
                  </div>
                </div>

                {/* File meta */}
                <div className={`flex items-center justify-between text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                  <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  <button onClick={reset} className="text-red-400 hover:text-red-500">
                    Remove
                  </button>
                </div>

                {/* Run button */}
                <Button
                  onClick={runAnalysis}
                  disabled={isAnalysing}
                  className="w-full bg-mint hover:bg-mint/90 text-navy font-semibold rounded-full h-12"
                >
                  {isAnalysing ? (
                    <span className="flex items-center gap-2">
                      <span className="w-4 h-4 border-2 border-navy/30 border-t-navy rounded-full animate-spin" />
                      Analysing…
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Brain className="w-5 h-5" />
                      Run Analysis
                    </span>
                  )}
                </Button>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-sm text-red-400">
                <span className="mt-0.5">⚠</span>
                {error}
              </div>
            )}
          </div>

          {/* ── Right: Results ── */}
          <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow flex flex-col gap-6`}>
            <h2 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>
              Analysis Results
            </h2>

            {!result && !isAnalysing && (
              <div className="flex-1 flex flex-col items-center justify-center py-16 text-center">
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${isDark ? 'bg-white/5' : 'bg-navy/5'}`}>
                  <Scan className={`w-8 h-8 ${isDark ? 'text-white/30' : 'text-navy/30'}`} />
                </div>
                <p className={isDark ? 'text-white/40' : 'text-navy/40'}>
                  Upload an image and run analysis to see results here
                </p>
              </div>
            )}

            {isAnalysing && (
              <div className="flex-1 flex flex-col items-center justify-center py-16 gap-4">
                <div className="w-12 h-12 border-4 border-mint/30 border-t-mint rounded-full animate-spin" />
                <p className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                  Running multi-disease analysis…
                </p>
              </div>
            )}

            {result && (
              <div className="space-y-5">

                {/* Attention map overlay */}
                <div className="relative rounded-2xl overflow-hidden h-52 bg-black cursor-pointer" onClick={() => setLightbox('anomaly')}>
                  {previewUrl && (
                    <img src={previewUrl} alt="Fundus" className="w-full h-full object-cover" />
                  )}
                  {/* Attention heatmap blended on top */}
                  <img
                    src={`data:image/png;base64,${result.attention_map_b64}`}
                    alt="Attention map"
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{ opacity: 0.85 }}
                  />
                  <div className="absolute top-3 right-3 px-3 py-1 bg-black/60 rounded-full text-xs text-white backdrop-blur-sm">
                    Anomaly map
                  </div>
                  <div className="absolute bottom-3 left-3 flex items-center gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${riskColour}`}>
                      {result.meta.risk_level} Risk
                    </span>
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-black/50 text-white backdrop-blur-sm">
                      Anomaly score: {Math.round(result.anomaly_score * 100)}%
                    </span>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/20">
                    <span className="text-white text-sm font-medium bg-black/50 px-4 py-2 rounded-full backdrop-blur-sm">Click to expand</span>
                  </div>
                </div>

                {/* Primary diagnosis */}
                <div className={`flex items-center justify-between p-4 rounded-2xl ${isDark ? 'bg-white/10' : 'bg-navy/5'}`}>
                  <div className="flex items-center gap-3">
                    <Eye className="w-5 h-5 text-mint" />
                    <span className={`font-medium text-sm ${isDark ? 'text-white' : 'text-navy'}`}>
                      Primary Diagnosis
                    </span>
                  </div>
                  <span className="text-mint font-bold text-sm">
                    {diseaseLabels[result.meta.primary_diagnosis]?.label ?? result.meta.primary_diagnosis}
                  </span>
                </div>

                {/* Per-disease probability bars */}
                <div className="space-y-3">
                  {(Object.keys(result.predictions) as Array<keyof typeof result.predictions>).map((key) => {
                    const d = result.predictions[key];
                    const meta = diseaseLabels[key];
                    return (
                      <ProbabilityBar
                        key={key}
                        label={meta?.label ?? key}
                        probability={d.probability}
                        severity={d.severity}
                        description={d.description}
                        color={meta?.color ?? 'text-mint'}
                        isDark={isDark}
                      />
                    );
                  })}
                </div>

                {/* Footer: scan meta */}
                <div className={`pt-3 border-t text-xs flex items-center justify-between ${isDark ? 'border-white/10 text-white/40' : 'border-navy/10 text-navy/40'}`}>
                  <span>Scan ID: {result.scan_id.slice(0, 8)}…</span>
                  <span>{result.inference_ms} ms</span>
                  <button
                    onClick={reset}
                    className="text-mint hover:underline"
                  >
                    New scan
                  </button>
                </div>

                {/* Discuss results with AI */}
                <Button
                  onClick={() => navigate('/app/chat')}
                  className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-full h-10"
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Discuss with AI
                </Button>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── Image Lightbox Modal ── */}
      {lightbox && <ImageLightbox
        type={lightbox}
        previewUrl={previewUrl}
        result={result}
        riskColour={riskColour}
        diseaseLabels={diseaseLabels}
        isDark={isDark}
        onClose={() => setLightbox(null)}
      />}
    </div>
  );
}

// ── Lightbox Component ────────────────────────────────────────────────────────

function ImageLightbox({
  type,
  previewUrl,
  result,
  riskColour,
  diseaseLabels,
  isDark,
  onClose,
}: {
  type: 'fundus' | 'anomaly';
  previewUrl: string | null;
  result: PredictResponse | null;
  riskColour: string;
  diseaseLabels: Record<string, { label: string; color: string }>;
  isDark: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className={`relative w-full max-w-6xl max-h-[90vh] rounded-3xl overflow-hidden flex flex-col lg:flex-row ${
          isDark ? 'bg-[#0d1a28]' : 'bg-white'
        } shadow-2xl`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-black/50 hover:bg-black/70 flex items-center justify-center text-white text-xl transition-colors backdrop-blur-sm"
        >
          ×
        </button>

        {/* Left: Full image — fills entire panel */}
        <div className="lg:w-3/5 w-full bg-black flex items-center justify-center relative min-h-[50vh] lg:min-h-0">
          {type === 'fundus' && previewUrl && (
            <img src={previewUrl} alt="Fundus full" className="w-full h-full object-contain" />
          )}
          {type === 'anomaly' && previewUrl && result && (
            <>
              <img src={previewUrl} alt="Fundus" className="w-full h-full object-contain" />
              <img
                src={`data:image/png;base64,${result.attention_map_b64}`}
                alt="Anomaly heatmap"
                className="absolute inset-0 w-full h-full object-contain"
                style={{ opacity: 0.75 }}
              />
            </>
          )}
        {/* Left: Full image */}
        <div className="lg:w-3/5 w-full bg-black flex items-center justify-center relative">
          <div className="relative inline-block">
            {type === 'fundus' && previewUrl && (
              <img src={previewUrl} alt="Fundus full" className="block max-w-full max-h-[85vh]" />
            )}
            {type === 'anomaly' && previewUrl && result && (
              <>
                <img src={previewUrl} alt="Fundus" className="block max-w-full max-h-[85vh]" />
                <img
                  src={`data:image/png;base64,${result.attention_map_b64}`}
                  alt="Anomaly heatmap"
                  className="absolute inset-0 w-full h-full"
                  style={{ opacity: 0.35, mixBlendMode: 'screen' }}
                />
              </>
            )}
          </div>
          <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/60 rounded-full text-xs text-white backdrop-blur-sm font-medium">
            {type === 'fundus' ? 'Original Fundus Image' : 'Anomaly Heatmap Overlay'}
          </div>
        </div>

        {/* Right: Details */}
        <div className={`lg:w-2/5 w-full p-6 lg:p-8 overflow-y-auto max-h-[90vh] flex flex-col gap-5 ${
          isDark ? 'text-white' : 'text-navy'
        }`}>
          <h3 className="text-xl font-bold">
            {type === 'fundus' ? 'Fundus Image Details' : 'Anomaly Map Details'}
          </h3>

          {result ? (
            <>
              {/* Risk + Anomaly */}
              <div className="flex items-center gap-3 flex-wrap">
                <span className={`px-4 py-1.5 rounded-full text-sm font-semibold ${riskColour}`}>
                  {result.meta.risk_level} Risk
                </span>
                <span className={`px-4 py-1.5 rounded-full text-sm font-medium ${
                  isDark ? 'bg-white/10 text-white/80' : 'bg-navy/10 text-navy/80'
                }`}>
                  Anomaly Score: {Math.round(result.anomaly_score * 100)}%
                </span>
              </div>

              {/* Primary diagnosis */}
              <div className={`p-4 rounded-2xl ${isDark ? 'bg-white/10' : 'bg-navy/5'}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Eye className="w-5 h-5 text-mint" />
                    <span className="font-medium text-sm">Primary Diagnosis</span>
                  </div>
                  <span className="text-mint font-bold text-sm">
                    {diseaseLabels[result.meta.primary_diagnosis]?.label ?? result.meta.primary_diagnosis}
                  </span>
                </div>
              </div>

              {/* Per-disease breakdown */}
              <div className="space-y-3">
                {(Object.keys(result.predictions) as Array<keyof typeof result.predictions>).map((key) => {
                  const d = result.predictions[key];
                  const meta = diseaseLabels[key];
                  return (
                    <ProbabilityBar
                      key={key}
                      label={meta?.label ?? key}
                      probability={d.probability}
                      severity={d.severity}
                      description={d.description}
                      color={meta?.color ?? 'text-mint'}
                      isDark={isDark}
                    />
                  );
                })}
              </div>

              {/* Scan metadata */}
              <div className={`pt-4 border-t text-xs space-y-1 ${isDark ? 'border-white/10 text-white/40' : 'border-navy/10 text-navy/40'}`}>
                <div className="flex justify-between"><span>Scan ID</span><span>{result.scan_id}</span></div>
                <div className="flex justify-between"><span>Inference Time</span><span>{result.inference_ms} ms</span></div>
                <div className="flex justify-between"><span>Model Confidence</span><span>{Math.round(result.meta.primary_probability * 100)}%</span></div>
              </div>
            </>
          ) : (
            <p className={isDark ? 'text-white/50' : 'text-navy/50'}>No analysis results yet. Run an analysis first.</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── AI Chat Page ──────────────────────────────────────────────────────────────

function ChatPage() {
  const { isDark } = useContext(DarkModeContext);
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [scanContext, setScanContext] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem('retinai_scan_context');
    if (stored) setScanContext(stored);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg = { role: 'user', content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
          scan_context: scanContext,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(body.detail);
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : 'Unable to connect. Make sure Ollama is running.';
      setMessages((prev) => [...prev, { role: 'assistant', content: msg }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const suggestions = [
    'What does my scan show?',
    'Explain diabetic retinopathy stages',
    'What treatments exist for glaucoma?',
    'How to prevent vision loss?',
  ];

  return (
    <div className={`h-screen flex flex-col ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />

      <main className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 min-h-0">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-navy'}`}>AI Assistant</h1>
            <p className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
              Powered by local LLM &mdash; your data stays on this machine
            </p>
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => {
                setMessages([]);
                setScanContext(sessionStorage.getItem('retinai_scan_context'));
              }}
              className="text-sm text-mint hover:underline"
            >
              New chat
            </button>
          )}
        </div>

        {/* Scan context badge */}
        {scanContext && (
          <div
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm mb-4 shrink-0 ${
              isDark ? 'bg-mint/10 text-mint' : 'bg-mint/10 text-emerald-700'
            }`}
          >
            <Eye className="w-4 h-4 shrink-0" />
            <span className="font-medium">Scan results attached</span>
            <span className={`hidden sm:inline ${isDark ? 'text-mint/60' : 'text-emerald-600/60'}`}>
              &mdash; AI can reference your latest analysis
            </span>
            <button
              onClick={() => {
                setScanContext(null);
                sessionStorage.removeItem('retinai_scan_context');
              }}
              className="ml-auto hover:underline text-xs"
            >
              Remove
            </button>
          </div>
        )}

        {/* Messages area */}
        <div
          className={`flex-1 overflow-y-auto rounded-3xl p-6 mb-4 min-h-0 ${
            isDark ? 'bg-white/5' : 'bg-white'
          } card-shadow`}
        >
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center py-12">
              <div className="w-16 h-16 rounded-2xl bg-mint/15 flex items-center justify-center mb-4">
                <Brain className="w-8 h-8 text-mint" />
              </div>
              <h3 className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
                RetinAI Assistant
              </h3>
              <p className={`text-sm mb-8 max-w-md ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                Ask about retinal scan results, eye diseases, treatment options, or anything ophthalmology-related.
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setInput(s);
                      inputRef.current?.focus();
                    }}
                    className={`px-4 py-2 rounded-full text-sm transition-colors ${
                      isDark
                        ? 'bg-white/10 text-white/70 hover:bg-white/15 hover:text-white'
                        : 'bg-navy/5 text-navy/70 hover:bg-navy/10 hover:text-navy'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-lg bg-mint/15 flex items-center justify-center mr-2 mt-1 shrink-0">
                      <Brain className="w-4 h-4 text-mint" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                      msg.role === 'user'
                        ? 'bg-mint text-navy rounded-br-md'
                        : isDark
                        ? 'bg-white/10 text-white/90 rounded-bl-md'
                        : 'bg-navy/5 text-navy rounded-bl-md'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="w-8 h-8 rounded-lg bg-mint/15 flex items-center justify-center mr-2 mt-1 shrink-0">
                    <Brain className="w-4 h-4 text-mint" />
                  </div>
                  <div className={`px-4 py-3 rounded-2xl rounded-bl-md ${isDark ? 'bg-white/10' : 'bg-navy/5'}`}>
                    <div className="flex gap-1.5">
                      <div
                        className={`w-2 h-2 rounded-full animate-bounce ${isDark ? 'bg-white/40' : 'bg-navy/40'}`}
                        style={{ animationDelay: '0ms' }}
                      />
                      <div
                        className={`w-2 h-2 rounded-full animate-bounce ${isDark ? 'bg-white/40' : 'bg-navy/40'}`}
                        style={{ animationDelay: '150ms' }}
                      />
                      <div
                        className={`w-2 h-2 rounded-full animate-bounce ${isDark ? 'bg-white/40' : 'bg-navy/40'}`}
                        style={{ animationDelay: '300ms' }}
                      />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className={`rounded-2xl p-3 flex items-end gap-3 shrink-0 ${isDark ? 'bg-white/10' : 'bg-white'} card-shadow`}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about retinal conditions, treatments, or your scan results\u2026"
            rows={1}
            className={`flex-1 resize-none bg-transparent outline-none text-sm px-2 py-2 max-h-32 ${
              isDark ? 'text-white placeholder:text-white/40' : 'text-navy placeholder:text-navy/40'
            }`}
            style={{ minHeight: '40px' }}
          />
          <Button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-mint hover:bg-mint/90 text-navy rounded-xl h-10 px-4 shrink-0"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </main>
    </div>
  );
}

// Patients Page — Coming Soon
function PatientsPage() {
  const { isDark } = useContext(DarkModeContext);

  const features = [
    {
      icon: Users,
      title: 'Patient Registry',
      description: 'Full patient database with demographics, contact info, and medical history',
    },
    {
      icon: History,
      title: 'Scan History',
      description: 'Complete timeline of all retinal scans per patient with side-by-side comparison',
    },
    {
      icon: Pill,
      title: 'Prescriptions',
      description: 'Generate and manage prescriptions directly from AI analysis results',
    },
    {
      icon: ClipboardList,
      title: 'Clinical Notes',
      description: 'Attach doctor notes, follow-up plans, and referral letters to each visit',
    },
    {
      icon: FolderOpen,
      title: 'PDF Reports',
      description: 'Export professional scan reports with heatmaps, probabilities, and recommendations',
    },
    {
      icon: Lock,
      title: 'HIPAA-Ready',
      description: 'Role-based access, audit logs, and encrypted storage for patient data compliance',
    },
  ];

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Coming Soon Banner */}
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-500/15 text-amber-500 text-sm font-semibold mb-6">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
            Coming Soon
          </div>
          <h1 className={`text-3xl sm:text-4xl font-bold mb-3 ${isDark ? 'text-white' : 'text-navy'}`}>
            Patient Management
          </h1>
          <p className={`text-lg max-w-xl mx-auto ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
            A full clinical workflow — from intake to prescription — all connected to your AI-powered retinal analysis.
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <div
              key={i}
              className={`group rounded-3xl p-6 transition-all duration-200 hover:scale-[1.02] ${
                isDark ? 'bg-white/5 hover:bg-white/8' : 'bg-white hover:shadow-lg'
              } card-shadow`}
            >
              <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-4 ${
                isDark ? 'bg-mint/10' : 'bg-mint/10'
              }`}>
                <f.icon className="w-6 h-6 text-mint" />
              </div>
              <h3 className={`font-semibold mb-1.5 ${isDark ? 'text-white' : 'text-navy'}`}>
                {f.title}
              </h3>
              <p className={`text-sm leading-relaxed ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                {f.description}
              </p>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className={`mt-12 text-center rounded-3xl p-8 ${
          isDark ? 'bg-white/5' : 'bg-white'
        } card-shadow`}>
          <Brain className={`w-10 h-10 mx-auto mb-3 ${isDark ? 'text-white/20' : 'text-navy/20'}`} />
          <p className={`text-sm ${isDark ? 'text-white/40' : 'text-navy/40'}`}>
            Patient management is under active development.<br />
            In the meantime, use <span className="text-mint font-medium">AI Chat</span> to discuss scan results and generate clinical insights.
          </p>
        </div>
      </main>
    </div>
  );
}

// Profile Page
function ProfilePage() {
  const { isDark } = useContext(DarkModeContext);

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />
      
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className={`text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
            My Profile
          </h1>
          <p className={`${isDark ? 'text-white/60' : 'text-navy/60'}`}>
            Manage your account and preferences
          </p>
        </div>

        <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="relative">
              <Avatar className="w-28 h-28 ring-4 ring-mint/30">
                <AvatarImage src="/avatar_02.jpg" />
                <AvatarFallback className="bg-mint/15 text-mint text-2xl">DR</AvatarFallback>
              </Avatar>
              <button className="absolute bottom-0 right-0 w-10 h-10 rounded-full bg-mint text-navy flex items-center justify-center shadow-lg hover:bg-mint/90 transition-colors">
                <Camera className="w-5 h-5" />
              </button>
            </div>
            
            <div className="text-center md:text-left">
              <h2 className={`text-xl font-bold mb-1 ${isDark ? 'text-white' : 'text-navy'}`}>
                Dr. David Roberts
              </h2>
              <p className={`mb-4 ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                Ophthalmologist
              </p>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-mint/15 text-mint rounded-full text-sm font-medium">
                Verified
              </div>
            </div>
          </div>

          <div className={`mt-8 pt-8 border-t ${isDark ? 'border-white/10' : 'border-navy/10'}`}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                { label: 'Full Name', value: '—' },
                { label: 'Email', value: '—' },
                { label: 'Phone', value: '—' },
                { label: 'License', value: '—' },
              ].map((field, i) => (
                <div key={i}>
                  <label className={`text-xs uppercase tracking-wider mb-1 block ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                    {field.label}
                  </label>
                  <p className={`font-medium ${isDark ? 'text-white' : 'text-navy'}`}>{field.value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Schedule Page
function SchedulePage() {
  const { isDark } = useContext(DarkModeContext);
  const now = new Date();
  const [currentMonth, setCurrentMonth] = useState(now.getMonth());
  const [currentYear, setCurrentYear] = useState(now.getFullYear());
  const [selectedDate, setSelectedDate] = useState(now.getDate());

  const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
  ];

  // Build calendar grid
  const firstDay = new Date(currentYear, currentMonth, 1);
  const lastDay = new Date(currentYear, currentMonth + 1, 0);
  const startDay = (firstDay.getDay() + 6) % 7; // Mon=0
  const totalDays = lastDay.getDate();
  const prevMonthLast = new Date(currentYear, currentMonth, 0).getDate();

  const cells: Array<{ day: number; inMonth: boolean }> = [];
  for (let i = startDay - 1; i >= 0; i--) cells.push({ day: prevMonthLast - i, inMonth: false });
  for (let d = 1; d <= totalDays; d++) cells.push({ day: d, inMonth: true });
  const remaining = 7 - (cells.length % 7);
  if (remaining < 7) for (let d = 1; d <= remaining; d++) cells.push({ day: d, inMonth: false });

  const isToday = (d: number) =>
    d === now.getDate() && currentMonth === now.getMonth() && currentYear === now.getFullYear();

  const prevMonth = () => {
    if (currentMonth === 0) { setCurrentMonth(11); setCurrentYear(currentYear - 1); }
    else setCurrentMonth(currentMonth - 1);
    setSelectedDate(0);
  };
  const nextMonth = () => {
    if (currentMonth === 11) { setCurrentMonth(0); setCurrentYear(currentYear + 1); }
    else setCurrentMonth(currentMonth + 1);
    setSelectedDate(0);
  };
  const goToday = () => {
    setCurrentMonth(now.getMonth());
    setCurrentYear(now.getFullYear());
    setSelectedDate(now.getDate());
  };

  // Sample appointments keyed by day-of-month
  const appointments: Record<number, Array<{ time: string; patient: string; type: string; color: string }>> = {
    [now.getDate()]: [
      { time: '09:00', patient: 'Sarah Johnson', type: 'DR Follow-up', color: 'bg-blue-500' },
      { time: '10:30', patient: 'Ahmed Hassan', type: 'Glaucoma Check', color: 'bg-purple-500' },
      { time: '14:00', patient: 'Maria Garcia', type: 'New Patient Scan', color: 'bg-mint' },
    ],
    [now.getDate() + 2]: [
      { time: '11:00', patient: 'James Lee', type: 'Post-Op Review', color: 'bg-amber-500' },
    ],
    [now.getDate() + 5]: [
      { time: '09:30', patient: 'Fatima Ali', type: 'Myopia Assessment', color: 'bg-red-500' },
      { time: '15:00', patient: 'Chen Wei', type: 'Annual Screening', color: 'bg-emerald-500' },
    ],
  };

  const daysWithAppts = new Set(Object.keys(appointments).map(Number));
  const todayAppts = appointments[selectedDate] ?? [];

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className={`text-2xl font-bold mb-1 ${isDark ? 'text-white' : 'text-navy'}`}>Clinic Schedule</h1>
            <p className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>Manage patient appointments &amp; follow-ups</p>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={goToday} className="rounded-full text-sm">
              Today
            </Button>
            <Button className="bg-mint hover:bg-mint/90 text-navy rounded-full">
              <Plus className="w-4 h-4 mr-2" />
              Book Appointment
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Calendar */}
          <div className={`lg:col-span-2 rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
            {/* Month nav */}
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <button onClick={prevMonth} className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors ${isDark ? 'hover:bg-white/10' : 'hover:bg-navy/5'}`}>
                  <ChevronLeft className={`w-5 h-5 ${isDark ? 'text-white/60' : 'text-navy/60'}`} />
                </button>
                <h2 className={`text-lg font-semibold min-w-[180px] text-center ${isDark ? 'text-white' : 'text-navy'}`}>
                  {monthNames[currentMonth]} {currentYear}
                </h2>
                <button onClick={nextMonth} className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors ${isDark ? 'hover:bg-white/10' : 'hover:bg-navy/5'}`}>
                  <ChevronRightIcon className={`w-5 h-5 ${isDark ? 'text-white/60' : 'text-navy/60'}`} />
                </button>
              </div>
            </div>

            {/* Day headers */}
            <div className="grid grid-cols-7 gap-1 mb-1">
              {dayNames.map((d) => (
                <div key={d} className={`text-center text-xs font-medium py-2 ${isDark ? 'text-white/40' : 'text-navy/40'}`}>{d}</div>
              ))}
            </div>

            {/* Date grid */}
            <div className="grid grid-cols-7 gap-1">
              {cells.map((c, i) => {
                const sel = c.inMonth && c.day === selectedDate;
                const today = c.inMonth && isToday(c.day);
                const hasAppt = c.inMonth && daysWithAppts.has(c.day);
                return (
                  <button
                    key={i}
                    onClick={() => c.inMonth && setSelectedDate(c.day)}
                    className={`relative aspect-square rounded-xl flex flex-col items-center justify-center text-sm transition-colors ${
                      sel
                        ? 'bg-mint text-navy font-bold shadow-md shadow-mint/25'
                        : today
                        ? isDark ? 'text-mint font-semibold bg-mint/10' : 'text-mint font-semibold bg-mint/10'
                        : c.inMonth
                        ? isDark ? 'hover:bg-white/10 text-white/80' : 'hover:bg-navy/5 text-navy/80'
                        : isDark ? 'text-white/15' : 'text-navy/15'
                    }`}
                  >
                    {c.day}
                    {hasAppt && !sel && (
                      <span className="absolute bottom-1.5 w-1.5 h-1.5 rounded-full bg-mint" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right sidebar: selected day */}
          <div className="space-y-6">
            {/* Day detail */}
            <div className={`rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
              <h3 className={`font-semibold mb-1 ${isDark ? 'text-white' : 'text-navy'}`}>
                {selectedDate > 0
                  ? `${monthNames[currentMonth]} ${selectedDate}`
                  : 'Select a date'
                }
              </h3>
              <p className={`text-xs mb-4 ${isDark ? 'text-white/40' : 'text-navy/40'}`}>
                {todayAppts.length} appointment{todayAppts.length !== 1 ? 's' : ''}
              </p>

              {todayAppts.length === 0 ? (
                <div className={`text-center py-8 ${isDark ? 'text-white/30' : 'text-navy/30'}`}>
                  <Calendar className="w-8 h-8 mx-auto mb-2 opacity-40" />
                  <p className="text-sm">No appointments</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {todayAppts.map((appt, i) => (
                    <div
                      key={i}
                      className={`flex items-start gap-3 p-3 rounded-2xl transition-colors ${
                        isDark ? 'bg-white/5 hover:bg-white/10' : 'bg-navy/5 hover:bg-navy/8'
                      }`}
                    >
                      <div className={`w-1 h-12 rounded-full shrink-0 ${appt.color}`} />
                      <div className="flex-1 min-w-0">
                        <p className={`text-sm font-semibold truncate ${isDark ? 'text-white' : 'text-navy'}`}>
                          {appt.patient}
                        </p>
                        <p className={`text-xs ${isDark ? 'text-white/50' : 'text-navy/50'}`}>{appt.type}</p>
                        <div className="flex items-center gap-1 mt-1">
                          <Clock className={`w-3 h-3 ${isDark ? 'text-white/30' : 'text-navy/30'}`} />
                          <span className={`text-xs ${isDark ? 'text-white/40' : 'text-navy/40'}`}>{appt.time}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quick stats */}
            <div className={`rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
              <h3 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy'}`}>This Week</h3>
              <div className="space-y-3">
                {[
                  { icon: Stethoscope, label: 'Consultations', value: '12', color: 'text-blue-500' },
                  { icon: Eye, label: 'Retinal Scans', value: '8', color: 'text-mint' },
                  { icon: Users, label: 'New Patients', value: '3', color: 'text-purple-500' },
                  { icon: ClipboardList, label: 'Follow-ups', value: '5', color: 'text-amber-500' },
                ].map((stat, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${isDark ? 'bg-white/5' : 'bg-navy/5'}`}>
                        <stat.icon className={`w-4 h-4 ${stat.color}`} />
                      </div>
                      <span className={`text-sm ${isDark ? 'text-white/70' : 'text-navy/70'}`}>{stat.label}</span>
                    </div>
                    <span className={`text-sm font-bold ${isDark ? 'text-white' : 'text-navy'}`}>{stat.value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Upcoming */}
            <div className={`rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
              <h3 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy'}`}>Upcoming</h3>
              <div className="space-y-3">
                {[
                  { patient: 'James Lee', type: 'Post-Op Review', days: 2 },
                  { patient: 'Fatima Ali', type: 'Myopia Assessment', days: 5 },
                  { patient: 'Chen Wei', type: 'Annual Screening', days: 5 },
                ].map((u, i) => (
                  <div key={i} className={`flex items-center justify-between py-2 ${
                    i < 2 ? (isDark ? 'border-b border-white/5' : 'border-b border-navy/5') : ''
                  }`}>
                    <div>
                      <p className={`text-sm font-medium ${isDark ? 'text-white/80' : 'text-navy/80'}`}>{u.patient}</p>
                      <p className={`text-xs ${isDark ? 'text-white/40' : 'text-navy/40'}`}>{u.type}</p>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      u.days <= 2 ? 'bg-amber-500/15 text-amber-500' : isDark ? 'bg-white/10 text-white/50' : 'bg-navy/5 text-navy/50'
                    }`}>
                      in {u.days}d
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Reports Page
function ReportsPage() {
  const { isDark } = useContext(DarkModeContext);

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className={`text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
              Reports
            </h1>
            <p className={`${isDark ? 'text-white/60' : 'text-navy/60'}`}>
              View patient analysis reports
            </p>
          </div>
          <Button variant="outline" className="rounded-full">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
        </div>

        <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
          <div className={`text-center py-12 ${isDark ? 'text-white/40' : 'text-navy/40'}`}>
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 ${isDark ? 'bg-white/5' : 'bg-navy/5'}`}>
              <FileText className={`w-8 h-8 ${isDark ? 'text-white/30' : 'text-navy/30'}`} />
            </div>
            <p>No reports available</p>
          </div>
        </div>
      </main>
    </div>
  );
}

// Settings Page
function SettingsPage() {
  const { isDark, toggleDark } = useContext(DarkModeContext);

  const settings = [
    { icon: User, title: 'Account', description: 'Manage your profile' },
    { icon: Bell, title: 'Notifications', description: 'Configure alerts' },
    { icon: Shield, title: 'Privacy', description: 'Security settings' },
    { icon: HelpCircle, title: 'Help', description: 'Get support' },
  ];

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />
      
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className={`text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
            Settings
          </h1>
          <p className={`${isDark ? 'text-white/60' : 'text-navy/60'}`}>
            Manage your preferences
          </p>
        </div>

        <div className="space-y-4">
          {/* Dark Mode Toggle */}
          <div className={`flex items-center justify-between p-6 rounded-2xl ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-mint/15 flex items-center justify-center">
                {isDark ? <Sun className="w-6 h-6 text-mint" /> : <Moon className="w-6 h-6 text-mint" />}
              </div>
              <div>
                <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>
                  Dark Mode
                </h3>
                <p className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                  Toggle between light and dark theme
                </p>
              </div>
            </div>
            <button
              onClick={toggleDark}
              className={`w-14 h-8 rounded-full transition-colors relative ${
                isDark ? 'bg-mint' : 'bg-navy/20'
              }`}
            >
              <div className={`absolute top-1 w-6 h-6 rounded-full bg-white transition-transform ${
                isDark ? 'translate-x-7' : 'translate-x-1'
              }`} />
            </button>
          </div>

          {/* Other Settings */}
          {settings.map((setting, i) => (
            <button
              key={i}
              className={`w-full flex items-center justify-between p-6 rounded-2xl transition-colors ${
                isDark ? 'bg-white/5 hover:bg-white/10' : 'bg-white hover:bg-navy/5'
              } card-shadow`}
            >
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${isDark ? 'bg-white/10' : 'bg-navy/5'}`}>
                  <setting.icon className={`w-6 h-6 ${isDark ? 'text-white/60' : 'text-navy/60'}`} />
                </div>
                <div className="text-left">
                  <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>
                    {setting.title}
                  </h3>
                  <p className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>
                    {setting.description}
                  </p>
                </div>
              </div>
              <ChevronRight className={`w-5 h-5 ${isDark ? 'text-white/30' : 'text-navy/30'}`} />
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}
