import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Button } from '@/components/ui/button';
import { ArrowRight, Mail } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

gsap.registerPlugin(ScrollTrigger);

export default function CTASection() {
  const sectionRef = useRef<HTMLElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const section = sectionRef.current;
    const content = contentRef.current;

    if (!section || !content) return;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        content.children,
        { y: 20, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          stagger: 0.08,
          duration: 0.6,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: content,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full bg-navy py-24 z-[110] overflow-hidden"
    >
      {/* Decorative Circle */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-mint/5" />

      <div className="max-w-[720px] mx-auto px-6 lg:px-12 relative">
        <div ref={contentRef} className="text-center">
          {/* Heading */}
          <h2 className="text-3xl lg:text-5xl font-bold text-white mb-6 leading-tight">
            Ready to detect eye diseases?
          </h2>

          {/* Body */}
          <p className="text-lg text-white/60 mb-10 max-w-lg mx-auto">
            Start using AI-powered retinal screening for faster, more accurate diagnoses.
          </p>

          {/* CTA - Single button for hackathon */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-10">
            <Button
              size="lg"
              className="bg-mint hover:bg-mint/90 text-navy px-8 rounded-full group"
              onClick={() => navigate('/app')}
            >
              Get Started
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>

          {/* Contact Line */}
          <div className="flex items-center justify-center gap-2 text-white/40 text-sm">
            <Mail className="w-4 h-4" />
            <span>Questions? Reach us at</span>
            <a href="mailto:hello@optiscan.ai" className="text-mint hover:underline">
              hello@optiscan.ai
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
