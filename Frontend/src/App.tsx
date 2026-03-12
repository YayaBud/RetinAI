import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import LandingPage from './pages/LandingPage';
import MainApp from './pages/MainApp';
import './App.css';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate initial loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-offwhite flex items-center justify-center z-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-mint/30 border-t-mint rounded-full animate-spin" />
          <span className="text-navy font-medium text-sm">Loading OptiScan AI...</span>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <AppContent />
    </Router>
  );
}

function AppContent() {
  const location = useLocation();
  const isLandingPage = location.pathname === '/';

  useEffect(() => {
    // Refresh ScrollTrigger on route change
    ScrollTrigger.refresh();
    
    // Scroll to top on route change
    window.scrollTo(0, 0);
  }, [location]);

  return (
    <div className={`min-h-screen ${isLandingPage ? 'bg-offwhite' : 'bg-offwhite'}`}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app/*" element={<MainApp />} />
      </Routes>
    </div>
  );
}

export default App;
