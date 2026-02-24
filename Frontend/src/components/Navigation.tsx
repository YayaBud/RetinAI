import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Eye, Menu, X } from 'lucide-react';

export default function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 100);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          isScrolled
            ? 'bg-offwhite/90 backdrop-blur-md shadow-sm'
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
              <span className="font-semibold text-navy text-lg tracking-tight">
                OptiScan AI
              </span>
            </Link>

            {/* Desktop Actions - Simplified for hackathon */}
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                className="text-navy/70 hover:text-navy hover:bg-navy/5"
                onClick={() => navigate('/app')}
              >
                Sign in
              </Button>
              <Button
                className="bg-navy hover:bg-navy/90 text-white px-6 rounded-full"
                onClick={() => navigate('/app')}
              >
                Get Started
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="lg:hidden p-2 rounded-lg hover:bg-navy/5 transition-colors"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6 text-navy" />
              ) : (
                <Menu className="w-6 h-6 text-navy" />
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu */}
      <div
        className={`fixed inset-0 z-40 lg:hidden transition-all duration-300 ${
          isMobileMenuOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
        }`}
      >
        <div
          className="absolute inset-0 bg-navy/20 backdrop-blur-sm"
          onClick={() => setIsMobileMenuOpen(false)}
        />
        <div
          className={`absolute top-16 left-4 right-4 bg-offwhite rounded-2xl card-shadow p-6 transition-all duration-300 ${
            isMobileMenuOpen ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'
          }`}
        >
          <div className="flex flex-col gap-4">
            <Button
              variant="ghost"
              className="justify-start text-navy/70 hover:text-navy"
              onClick={() => {
                setIsMobileMenuOpen(false);
                navigate('/app');
              }}
            >
              Sign in
            </Button>
            <Button
              className="bg-navy hover:bg-navy/90 text-white rounded-full"
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
