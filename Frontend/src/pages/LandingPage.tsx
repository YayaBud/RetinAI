import Navigation from '../components/Navigation';
import HeroSection from '../sections/HeroSection';
import PipelineSection from '../sections/PipelineSection';
import UploadSection from '../sections/UploadSection';
import AnalysisSection from '../sections/AnalysisSection';
import ResultsSection from '../sections/ResultsSection';
import DashboardSection from '../sections/DashboardSection';
import CalendarSection from '../sections/CalendarSection';
import PatientsSection from '../sections/PatientsSection';
import StatsSection from '../sections/StatsSection';
import TestimonialsSection from '../sections/TestimonialsSection';
import CTASection from '../sections/CTASection';

export default function LandingPage() {
  return (
    <div className="relative bg-[#080F17]">
      <Navigation />

      {/* Floating cell-like particles */}
      <div className="eye-particles" aria-hidden="true" />

      {/* Radial pulse rings — iris scan sonar effect */}
      <div className="eye-pulse" aria-hidden="true">
        <div className="eye-pulse-ring" />
        <div className="eye-pulse-ring" style={{ animationDelay: '2s' }} />
        <div className="eye-pulse-ring" style={{ animationDelay: '4s' }} />
        <div className="eye-pulse-ring" style={{ animationDelay: '6s' }} />
      </div>

      {/* Scanning grid background */}
      <div className="scan-grid" aria-hidden="true" />

      {/* Horizontal scan line */}
      <div className="scan-line" aria-hidden="true" />

      <main className="relative">
        <HeroSection />
        <PipelineSection />
        <UploadSection />
        <AnalysisSection />
        <ResultsSection />
        <DashboardSection />
        <CalendarSection />
        <PatientsSection />
        <StatsSection />
        <TestimonialsSection />
        <CTASection />
      </main>
    </div>
  );
}
