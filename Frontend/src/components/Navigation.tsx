import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Microscope, Menu, X } from 'lucide-react';

export default function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 60);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { label: 'ACCURACY', href: '#accuracy' },
    { label: 'FEATURES', href: '#features' },
    { label: 'CASE STUDIES', href: '#architecture' },
    { label: 'PROTOCOLS', href: '#cta' },
  ];

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          isScrolled
            ? 'backdrop-blur-xl'
            : 'bg-transparent'
        }`}
        style={isScrolled ? { background: 'rgba(47,89,104,0.9)' } : undefined}
      >
        <div className="w-full max-w-[1400px] mx-auto px-6 lg:px-12">
          <div className="flex items-center justify-between h-16 lg:h-20">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 group">
              <Microscope className="w-6 h-6 text-[#27D17F]" />
              <span
                className="font-bold text-white text-lg tracking-tight"
                style={{ fontFamily: "'Space Grotesk', sans-serif" }}
              >
                Retina AI
              </span>
            </Link>

            {/* Desktop Nav Links */}
            <div className="hidden lg:flex items-center gap-8">
              {navLinks.map((link, i) => (
                <a
                  key={link.label}
                  href={link.href}
                  className={`text-[11px] font-medium tracking-[0.18em] transition-colors ${
                    i === 0
                      ? 'text-white border-b border-white/40 pb-0.5'
                      : 'text-white/50 hover:text-white/80'
                  }`}
                >
                  {link.label}
                </a>
              ))}
            </div>

            {/* Desktop CTA */}
            <div className="hidden lg:flex items-center">
              <Button
                className="bg-[#27D17F] hover:bg-[#22b86e] text-[#0a1e2c] px-7 rounded-xl font-semibold text-sm shadow-lg shadow-[#27D17F]/15"
                onClick={() => navigate('/app')}
              >
                GET STARTED
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button
              className="lg:hidden p-2 rounded-lg hover:bg-white/10 transition-colors"
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

      {/* Mobile Menu */}
      <div
        className={`fixed inset-0 z-40 lg:hidden transition-all duration-300 ${
          isMobileMenuOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
        }`}
      >
        <div
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={() => setIsMobileMenuOpen(false)}
        />
        <div
          className={`absolute top-16 left-4 right-4 rounded-2xl p-6 transition-all duration-300 ${
            isMobileMenuOpen ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0'
          }`}
          style={{ background: '#2a5160', boxShadow: '0 20px 40px rgba(0,0,0,0.3)' }}
        >
          <div className="flex flex-col gap-3">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="text-xs font-medium tracking-[0.15em] text-white/60 hover:text-white py-2 transition-colors"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                {link.label}
              </a>
            ))}
            <Button
              className="bg-[#27D17F] hover:bg-[#22b86e] text-[#0a1e2c] rounded-xl font-semibold mt-2"
              onClick={() => {
                setIsMobileMenuOpen(false);
                navigate('/app');
              }}
            >
              GET STARTED
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
