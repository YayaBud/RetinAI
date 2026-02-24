import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Eye, Layers, Scan, CircleDot } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function AnalysisSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [activeOverlay, setActiveOverlay] = useState('lesions');

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const overlay = overlayRef.current;

    if (!section || !leftCard || !rightCard || !overlay) return;

    const ctx = gsap.context(() => {
      const scrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: 'top top',
          end: '+=100%',
          pin: true,
          scrub: 0.4,
        },
      });

      // ENTRANCE (0-40%)
      scrollTl
        .fromTo(leftCard, { x: '-50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(rightCard, { x: '50vw', opacity: 0 }, { x: 0, opacity: 1, ease: 'none' }, 0)
        .fromTo(
          overlay,
          { opacity: 0, scale: 0.98 },
          { opacity: 0.65, scale: 1, ease: 'none' },
          0.08
        );

      // EXIT (60-100%)
      scrollTl
        .fromTo(leftCard, { x: 0, opacity: 1 }, { x: '-8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(rightCard, { x: 0, opacity: 1 }, { x: '8vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(overlay, { opacity: 0.65 }, { opacity: 0, ease: 'power2.in' }, 0.6);
    }, section);

    return () => ctx.revert();
  }, []);

  const overlays = [
    { id: 'vessels', icon: Scan, label: 'Vessels' },
    { id: 'lesions', icon: CircleDot, label: 'Lesions' },
    { id: 'optic', icon: Eye, label: 'Optic Disc' },
  ];

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-40"
    >
      {/* Left Media Card */}
      <div
        ref={leftCardRef}
        className="absolute rounded-[28px] overflow-hidden card-shadow"
        style={{
          left: '10vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <img
          src="/analysis_retina_closeup.jpg"
          alt="Retina closeup"
          className="w-full h-full object-cover"
        />

        {/* Attention Overlay */}
        <div
          ref={overlayRef}
          className="absolute inset-0 animate-pulse-soft"
          style={{
            background: `radial-gradient(ellipse at 60% 45%, rgba(39, 209, 127, 0.4) 0%, rgba(39, 209, 127, 0.15) 30%, transparent 60%)`,
            mixBlendMode: 'overlay',
          }}
        />

        {/* Heatmap Points */}
        <div className="absolute inset-0">
          <div
            className="absolute w-4 h-4 rounded-full bg-mint/80 animate-pulse"
            style={{ top: '40%', left: '55%' }}
          />
          <div
            className="absolute w-3 h-3 rounded-full bg-mint/60 animate-pulse"
            style={{ top: '50%', left: '60%', animationDelay: '0.5s' }}
          />
          <div
            className="absolute w-2 h-2 rounded-full bg-mint/50 animate-pulse"
            style={{ top: '45%', left: '50%', animationDelay: '1s' }}
          />
        </div>
      </div>

      {/* Right Content Card */}
      <div
        ref={rightCardRef}
        className="absolute bg-offwhite rounded-[28px] card-shadow flex flex-col justify-center p-10"
        style={{
          left: '52vw',
          top: '18vh',
          width: '38vw',
          height: '64vh',
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-mint" />
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
            AI Visualization
          </span>
        </div>

        <h2 className="text-3xl lg:text-4xl font-bold text-navy mb-4 leading-tight">
          See what the AI sees
        </h2>

        <p className="text-base text-navy/60 leading-relaxed max-w-md mb-8">
          Attention maps highlight regions that influenced the prediction—so you can verify before you decide.
        </p>

        {/* Toggle Overlays */}
        <div className="space-y-3">
          <p className="text-sm text-navy/50 mb-3">Toggle overlays:</p>
          <div className="flex gap-3">
            {overlays.map((overlay) => (
              <button
                key={overlay.id}
                onClick={() => setActiveOverlay(overlay.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-all ${
                  activeOverlay === overlay.id
                    ? 'bg-mint text-navy'
                    : 'bg-navy/5 text-navy/60 hover:bg-navy/10'
                }`}
              >
                <overlay.icon className="w-4 h-4" />
                {overlay.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
