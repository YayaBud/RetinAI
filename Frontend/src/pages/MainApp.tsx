import { useState, createContext, useContext } from 'react';
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
    {
      id: 'cataract',
      name: 'Cataract',
      description: 'Detect lens clouding and opacity',
      icon: Scan,
      color: 'from-emerald-500/20 to-emerald-600/10',
      iconColor: 'text-emerald-500',
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
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

// Health Page - Upload and Analysis
function HealthPage() {
  const { isDark } = useContext(DarkModeContext);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const handleUpload = () => {
    setIsUploading(true);
    setUploadProgress(0);
    setAnalysisComplete(false);

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          setAnalysisComplete(true);
          return 100;
        }
        return prev + 5;
      });
    }, 100);
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
            Upload retinal images for AI-powered disease detection
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Area */}
          <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
            <h2 className={`text-lg font-semibold mb-6 ${isDark ? 'text-white' : 'text-navy'}`}>
              Upload Image
            </h2>

            {!isUploading && !analysisComplete && (
              <div
                onClick={handleUpload}
                className={`border-2 border-dashed border-mint/40 rounded-2xl p-12 text-center cursor-pointer transition-colors ${
                  isDark ? 'hover:bg-white/5' : 'hover:bg-mint/5'
                }`}
              >
                <div className="w-16 h-16 rounded-2xl bg-mint/15 flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-8 h-8 text-mint" />
                </div>
                <p className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
                  Click to upload or drag and drop
                </p>
                <p className={`text-sm ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                  Supports DICOM, JPEG, PNG (max 50MB)
                </p>
              </div>
            )}

            {isUploading && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-sm">
                  <span className={isDark ? 'text-white/70' : 'text-navy/70'}>Uploading image...</span>
                  <span className="text-mint font-semibold">{uploadProgress}%</span>
                </div>
                <div className="h-3 bg-navy/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-mint rounded-full transition-all duration-100"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {analysisComplete && (
              <div className="text-center">
                <div className="w-16 h-16 rounded-2xl bg-mint/15 flex items-center justify-center mx-auto mb-4">
                  <Activity className="w-8 h-8 text-mint" />
                </div>
                <p className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
                  Analysis Complete!
                </p>
                <p className={`text-sm mb-4 ${isDark ? 'text-white/50' : 'text-navy/50'}`}>
                  View results on the right
                </p>
                <Button 
                  onClick={() => { setAnalysisComplete(false); }} 
                  variant="outline" 
                  className="rounded-full"
                >
                  Upload Another
                </Button>
              </div>
            )}
          </div>

          {/* Analysis Results */}
          <div className={`rounded-3xl p-8 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
            <h2 className={`text-lg font-semibold mb-6 ${isDark ? 'text-white' : 'text-navy'}`}>
              Analysis Results
            </h2>

            {analysisComplete ? (
              <div className="space-y-6">
                {/* Retina Image Preview */}
                <div className="relative rounded-xl overflow-hidden">
                  <img
                    src="/analysis_retina_closeup.jpg"
                    alt="Retina scan"
                    className="w-full h-48 object-cover"
                  />
                  <div
                    className="absolute inset-0 animate-pulse-soft"
                    style={{
                      background: `radial-gradient(ellipse at 60% 45%, rgba(39, 209, 127, 0.3) 0%, transparent 60%)`,
                      mixBlendMode: 'overlay',
                    }}
                  />
                  <div className="absolute top-3 right-3 px-3 py-1 bg-mint rounded-full text-xs font-semibold text-navy">
                    AI Analyzed
                  </div>
                </div>

                {/* Findings */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 bg-mint/10 rounded-xl">
                    <div className="flex items-center gap-3">
                      <Eye className="w-5 h-5 text-mint" />
                      <span className={`font-medium ${isDark ? 'text-white' : 'text-navy'}`}>Detected Condition</span>
                    </div>
                    <span className="text-mint font-semibold">—</span>
                  </div>
                </div>

                {/* Confidence */}
                <div className={`pt-4 border-t ${isDark ? 'border-white/10' : 'border-navy/10'}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm ${isDark ? 'text-white/60' : 'text-navy/60'}`}>AI Confidence</span>
                    <span className="text-sm font-semibold text-mint">—</span>
                  </div>
                  <div className="h-2 bg-navy/10 rounded-full overflow-hidden">
                    <div className="h-full bg-mint rounded-full" style={{ width: '0%' }} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 ${isDark ? 'bg-white/5' : 'bg-navy/5'}`}>
                  <Scan className={`w-8 h-8 ${isDark ? 'text-white/30' : 'text-navy/30'}`} />
                </div>
                <p className={isDark ? 'text-white/50' : 'text-navy/50'}>
                  Upload an image to see analysis results
                </p>
              </div>
            )}
          </div>
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
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const dates = [15, 16, 17, 18, 19, 20, 21];

  return (
    <div className={`min-h-screen ${isDark ? 'bg-navy' : 'bg-offwhite'}`}>
      <TopNav />
      
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className={`text-2xl font-bold mb-2 ${isDark ? 'text-white' : 'text-navy'}`}>
              Schedule
            </h1>
            <p className={`${isDark ? 'text-white/60' : 'text-navy/60'}`}>
              Manage your appointments
            </p>
          </div>
          <Button className="bg-mint hover:bg-mint/90 text-navy rounded-full">
            <Plus className="w-4 h-4 mr-2" />
            New Appointment
          </Button>
        </div>

        <div className={`rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
          {/* Calendar Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <button className={`w-10 h-10 rounded-xl flex items-center justify-center ${isDark ? 'hover:bg-white/10' : 'hover:bg-navy/5'}`}>
                <ChevronLeft className={`w-5 h-5 ${isDark ? 'text-white/60' : 'text-navy/60'}`} />
              </button>
              <span className={`font-semibold ${isDark ? 'text-white' : 'text-navy'}`}>January 2025</span>
              <button className={`w-10 h-10 rounded-xl flex items-center justify-center ${isDark ? 'hover:bg-white/10' : 'hover:bg-navy/5'}`}>
                <ChevronRightIcon className={`w-5 h-5 ${isDark ? 'text-white/60' : 'text-navy/60'}`} />
              </button>
            </div>
          </div>

          {/* Days */}
          <div className="grid grid-cols-7 gap-2 mb-2">
            {days.map((day) => (
              <div key={day} className={`text-center text-xs py-2 ${isDark ? 'text-white/40' : 'text-navy/40'}`}>
                {day}
              </div>
            ))}
          </div>

          {/* Dates */}
          <div className="grid grid-cols-7 gap-2">
            {dates.map((date, i) => (
              <button
                key={date}
                className={`aspect-square rounded-xl flex items-center justify-center text-sm transition-colors ${
                  i === 3
                    ? 'bg-mint text-navy font-semibold'
                    : isDark
                    ? 'hover:bg-white/10 text-white/70'
                    : 'hover:bg-navy/5 text-navy/70'
                }`}
              >
                {date}
              </button>
            ))}
          </div>
        </div>

        {/* Appointments */}
        <div className={`mt-6 rounded-3xl p-6 ${isDark ? 'bg-white/5' : 'bg-white'} card-shadow`}>
          <h3 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-navy'}`}>
            Today's Appointments
          </h3>
          <div className={`text-center py-8 ${isDark ? 'text-white/40' : 'text-navy/40'}`}>
            No appointments scheduled
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
