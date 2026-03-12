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
      gsap.set([leftCard, rightCard], { opacity: 0, scale: 0.85 });
      gsap.set(overlay, { opacity: 0 });

      // Fade out as next section scrolls over
      gsap.to(section, {
        opacity: 0,
        y: -40,
        scrollTrigger: {
          trigger: section,
          start: 'bottom bottom',
          end: 'bottom 20%',
          scrub: 0.3,
        },
      });

      ScrollTrigger.create({
        trigger: section,
        start: 'top 70%',
        onEnter: () => {
          const tl = gsap.timeline({ defaults: { ease: 'power2.out', duration: 0.7 } });
          tl.to(leftCard, { scale: 1, opacity: 1 }, 0)
            .to(rightCard, { scale: 1, opacity: 1 }, 0.1)
            .to(overlay, { opacity: 0.65 }, 0.2);
        },
        once: true,
      });
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
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 4, backgroundColor: 'rgba(8,15,23,0.55)' }}
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
        className="absolute bg-[#0d1a28] rounded-[28px] card-shadow border border-white/5 flex flex-col justify-center p-10"
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

        <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4 leading-tight">
          See what the AI sees
        </h2>

        <p className="text-base text-white/50 leading-relaxed max-w-md mb-8">
          Attention maps highlight regions that influenced the prediction—so you can verify before you decide.
        </p>

        {/* Toggle Overlays */}
        <div className="space-y-3">
          <p className="text-sm text-white/40 mb-3">Toggle overlays:</p>
          <div className="flex gap-3">
            {overlays.map((overlay) => (
              <button
                key={overlay.id}
                onClick={() => setActiveOverlay(overlay.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-all ${
                  activeOverlay === overlay.id
                    ? 'bg-mint text-navy'
                    : 'bg-white/5 text-white/50 hover:bg-white/10'
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
