import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Eye, Menu, X } from 'lucide-react';

export default function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrollPct, setScrollPct] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100);
      const el  = document.documentElement;
      const pct = el.scrollTop / (el.scrollHeight - el.clientHeight);
      setScrollPct(isNaN(pct) ? 0 : Math.min(pct * 100, 100));
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          isScrolled
            ? 'bg-[#080F17]/90 backdrop-blur-md border-b border-white/5'
            : 'bg-transparent'
        }`}
      >
        <div className="w-full px-6 lg:px-12">
          <div className="flex items-center justify-between h-16 lg:h-20">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2 group">
              <div className="w-9 h-9 rounded-xl bg-mint/15 flex items-center justify-center group-hover:bg-mint/25 transition-colors">
                <Eye className="w-5 h-5 text-mint" />
              </div>
              <span className="font-semibold text-white text-lg tracking-tight">
                OptiScan AI
              </span>
            </Link>

            {/* Desktop Actions */}
            <div className="hidden lg:flex items-center gap-4">
              <Button
                variant="ghost"
                className="text-white/70 hover:text-white hover:bg-white/5"
                onClick={() => navigate('/app')}
              >
                Sign in
              </Button>
              <Button
                className="bg-mint hover:bg-mint/90 text-[#080F17] px-6 rounded-full font-semibold"
                onClick={() => navigate('/app')}
              >
                Get Started
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="lg:hidden p-2 rounded-lg hover:bg-white/5 transition-colors"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6 text-white" />
              ) : (
                <Menu className="w-6 h-6 text-white" />
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* Scroll progress bar */}
      <div className="fixed top-0 left-0 right-0 z-[100] h-[2px] pointer-events-none">
        <div
          className="h-full bg-mint transition-none"
          style={{ width: `${scrollPct}%`, boxShadow: '0 0 8px rgba(39,209,127,0.6)' }}
        />
      </div>

      {/* Mobile Menu */}
      <div
        className={`fixed inset-0 z-40 lg:hidden transition-all duration-300 ${
          isMobileMenuOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
        }`}
      >
        <div
          className="absolute inset-0 bg-black/40 backdrop-blur-sm"
          onClick={() => setIsMobileMenuOpen(false)}
        />
        <div
          className={`absolute top-16 left-4 right-4 bg-[#0d1a28] rounded-2xl border border-white/10 p-6 transition-all duration-300 ${
            isMobileMenuOpen ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'
          }`}
          style={{ boxShadow: '0 18px 50px rgba(0,0,0,0.4)' }}
        >
          <div className="flex flex-col gap-4">
            <Button
              variant="ghost"
              className="justify-start text-white/70 hover:text-white"
              onClick={() => {
                setIsMobileMenuOpen(false);
                navigate('/app');
              }}
            >
              Sign in
            </Button>
            <Button
              className="bg-mint hover:bg-mint/90 text-[#080F17] rounded-full font-semibold"
              onClick={() => {
                setIsMobileMenuOpen(false);
                navigate('/app');
              }}
            >
              Get Started
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
