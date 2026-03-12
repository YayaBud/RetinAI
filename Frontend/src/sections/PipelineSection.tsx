import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Upload, Brain, FileText, Zap } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function PipelineSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const chipsRef = useRef<HTMLDivElement>(null);
  const miniCardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const chips = chipsRef.current;
    const miniCard = miniCardRef.current;

    if (!section || !leftCard || !rightCard || !chips || !miniCard) return;

    const ctx = gsap.context(() => {
      // Set initial hidden state
      gsap.set([leftCard, rightCard, miniCard], { opacity: 0 });
      gsap.set(leftCard, { y: -80 });
      gsap.set(rightCard, { y: 80 });
      gsap.set(chips.children, { opacity: 0, y: -30 });
      gsap.set(miniCard, { scale: 0.6, y: 40 });

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
          tl.to(leftCard, { y: 0, opacity: 1 }, 0)
            .to(rightCard, { y: 0, opacity: 1 }, 0.08)
            .to(chips.children, { y: 0, opacity: 1, stagger: 0.06 }, 0.15)
            .to(miniCard, { scale: 1, y: 0, opacity: 1, ease: 'back.out(1.4)' }, 0.2);
        },
        once: true,
      });
    }, section);

    return () => ctx.revert();
  }, []);

  const steps = [
    { icon: Upload, label: 'Upload', active: true },
    { icon: Brain, label: 'AI Analysis', active: false },
    { icon: FileText, label: 'Report', active: false },
  ];

  return (
    <section
      ref={sectionRef}
      id="pipeline"
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: 'rgba(8,15,23,0.55)' }}
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
          src="/pipeline_retina_scan.jpg"
          alt="Retina scan on monitor"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-navy/10 to-transparent" />
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
        {/* Step Chips */}
        <div ref={chipsRef} className="flex gap-3 mb-8">
          {steps.map((step) => (
            <div
              key={step.label}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-all ${
                step.active
                  ? 'bg-mint text-navy'
                  : 'bg-white/5 text-white/50 hover:bg-white/10'
              }`}
            >
              <step.icon className="w-4 h-4" />
              {step.label}
            </div>
          ))}
        </div>

        {/* Title */}
        <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4 leading-tight">
          From image to insight—in seconds
        </h2>

        {/* Body */}
        <p className="text-base text-white/50 leading-relaxed max-w-md">
          Our AI pipeline detects anomalies, generates attention maps, and classifies risk across four major eye conditions.
        </p>
      </div>

      {/* Mini Card */}
      <div
        ref={miniCardRef}
        className="absolute bg-[#0d1a28] rounded-[22px] card-shadow border border-white/5 p-6 flex flex-col justify-center"
        style={{
          right: '8vw',
          bottom: '10vh',
          width: '18vw',
          height: '18vh',
          minWidth: '200px',
        }}
      >
        <div className="flex items-center gap-2 text-white/40 text-sm mb-2">
          <Zap className="w-4 h-4" />
          Typical scan time
        </div>
        <div className="text-4xl font-bold text-mint">&lt; 4 sec</div>
      </div>
    </section>
  );
}
