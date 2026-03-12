import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Button } from '@/components/ui/button';
import { ArrowRight, Activity, Brain, Scan } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

gsap.registerPlugin(ScrollTrigger);

export default function HeroSection() {
  const sectionRef  = useRef<HTMLElement>(null);
  const bgRef       = useRef<HTMLDivElement>(null);
  const eyeRef      = useRef<HTMLDivElement>(null);
  const ring1Ref    = useRef<HTMLDivElement>(null);
  const ring2Ref    = useRef<HTMLDivElement>(null);
  const ring3Ref    = useRef<HTMLDivElement>(null);
  const scanRef     = useRef<HTMLDivElement>(null);
  const dotsRef     = useRef<HTMLDivElement>(null);
  const textRef     = useRef<HTMLDivElement>(null);
  const statsRef    = useRef<HTMLDivElement>(null);
  const navigate    = useNavigate();

  useEffect(() => {
    const ctx = gsap.context(() => {
      // ── Initial invisibility ──────────────────────────────────────────────
      gsap.set([eyeRef.current, ring1Ref.current, ring2Ref.current, ring3Ref.current,
                scanRef.current], { opacity: 0 });
      gsap.set(eyeRef.current,   { scale: 0.78 });
      gsap.set(ring1Ref.current, { scale: 0.55 });
      gsap.set(ring2Ref.current, { scale: 0.40 });
      gsap.set(ring3Ref.current, { scale: 0.25 });
      gsap.set(textRef.current?.children  ?? [], { opacity: 0, y: 56 });
      gsap.set(statsRef.current?.children ?? [], { opacity: 0, x: 60 });
      if (dotsRef.current?.children) {
        gsap.set(dotsRef.current.children, { opacity: 0, scale: 0 });
      }

      // ── Load animation ────────────────────────────────────────────────────
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

      tl.fromTo(bgRef.current, { opacity: 0 }, { opacity: 1, duration: 0.7 })
        .to(eyeRef.current,    { opacity: 1, scale: 1, duration: 1.05, ease: 'back.out(1.15)' }, 0.25)
        .to(ring1Ref.current,  { opacity: 1, scale: 1, duration: 1.4,  ease: 'elastic.out(1, 0.60)' }, 0.40)
        .to(ring2Ref.current,  { opacity: 0.6, scale: 1, duration: 1.6, ease: 'elastic.out(1, 0.55)' }, 0.50)
        .to(ring3Ref.current,  { opacity: 0.3, scale: 1, duration: 1.8, ease: 'elastic.out(1, 0.50)' }, 0.58)
        .to(textRef.current?.children  ?? [], { opacity: 1, y: 0, stagger: 0.11, duration: 0.7 }, 0.65)
        .to(statsRef.current?.children ?? [], { opacity: 1, x: 0, stagger: 0.13, duration: 0.6 }, 0.80)
        .to(scanRef.current, { opacity: 1, duration: 0.4 }, 1.0);

      if (dotsRef.current?.children) {
        tl.to(dotsRef.current.children, {
          opacity: 1, scale: 1, stagger: 0.25, duration: 0.5, ease: 'back.out(2)',
        }, 1.1);
      }

      // ── Ambient ring rotations ────────────────────────────────────────────
      gsap.to(ring1Ref.current, { rotate: 360,  duration: 22, repeat: -1, ease: 'none' });
      gsap.to(ring2Ref.current, { rotate: -360, duration: 34, repeat: -1, ease: 'none' });
      gsap.to(ring3Ref.current, { rotate: 180,  duration: 50, repeat: -1, ease: 'none' });

      // ── Scan-line sweep loop ───────────────────────────────────────────────
      gsap.fromTo(scanRef.current,
        { top: '10%' },
        { top: '90%', duration: 2.8, repeat: -1, yoyo: true, ease: 'sine.inOut', delay: 1.2 }
      );

      // ── Fade out as next section scrolls over ────────────────────────────
      gsap.to(sectionRef.current, {
        opacity: 0,
        y: -40,
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'bottom bottom',
          end: 'bottom 20%',
          scrub: 0.3,
        },
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  const stats = [
    { icon: Activity, label: 'Test Accuracy',  value: '93.82%',   sub: 'EfficientNet-B3' },
    { icon: Brain,    label: 'Disease Types',  value: '3 Classes', sub: 'DR · Glaucoma · Myopia' },
    { icon: Scan,     label: 'Input Size',     value: '256 px',    sub: 'Diffusion model' },
  ];

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden flex items-center justify-center"
      style={{ position: 'sticky', top: 0, zIndex: 1, backgroundColor: 'rgba(8,15,23,0.55)' }}
    >
      {/* ── Background ─────────────────────────────────────────────────────── */}
      <div ref={bgRef} className="absolute inset-0 opacity-0">
        <div className="absolute inset-0" style={{ background: 'rgba(8,15,23,0.5)' }} />
        {/* Ambient glow */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none"
          style={{
            width: '820px', height: '820px',
            background: 'radial-gradient(circle, rgba(39,209,127,0.09) 0%, rgba(39,209,127,0.03) 45%, transparent 70%)',
          }}
        />
        {/* Dot grid */}
        <div className="absolute inset-0 dot-pattern opacity-[0.13]" />
        {/* Edge vignette */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: 'radial-gradient(ellipse at center, transparent 38%, rgba(8,15,23,0.65) 100%)' }}
        />
      </div>

      {/* ── Three-column layout ────────────────────────────────────────────── */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-10 flex items-center gap-10 lg:gap-14">

        {/* LEFT: Text ──────────────────────────────────────────────────────── */}
        <div ref={textRef} className="flex-1 min-w-0 flex flex-col gap-5">
          {/* Badge */}
          <div
            className="flex items-center gap-2.5 w-fit px-4 py-2 rounded-full"
            style={{ border: '1px solid rgba(39,209,127,0.22)', background: 'rgba(39,209,127,0.07)' }}
          >
            <span className="w-2 h-2 rounded-full bg-mint animate-pulse flex-none" />
            <span className="text-mint text-[11px] font-semibold tracking-[0.18em] uppercase whitespace-nowrap">
              AI-Powered Retinal Screening
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-[clamp(2.4rem,4.2vw,3.6rem)] font-bold leading-[1.07] text-white">
            Retinal<br />
            <span className="text-mint">Intelligence</span>
          </h1>

          {/* Body */}
          <p className="text-white/52 text-base lg:text-lg leading-relaxed max-w-[26rem]">
            Three EfficientNet-B3 models fused with a diffusion-based anomaly
            detector for simultaneous multi-disease classification.
          </p>

          {/* CTA */}
          <div className="flex items-center gap-5 pt-1">
            <Button
              size="lg"
              className="bg-mint hover:bg-mint/90 text-[#080F17] font-bold px-8 rounded-full group"
              style={{ boxShadow: '0 0 28px rgba(39,209,127,0.22)' }}
              onClick={() => navigate('/app')}
            >
              Start Scanning
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
            <span className="text-white/25 text-sm">No login required</span>
          </div>
        </div>

        {/* CENTRE: Retinal eye ─────────────────────────────────────────────── */}
        <div className="relative flex-none w-[clamp(220px,24vw,300px)] h-[clamp(220px,24vw,300px)]">
          {/* Outer dashed ring */}
          <div
            ref={ring3Ref}
            className="absolute inset-0 rounded-full opacity-0"
            style={{ border: '1px dashed rgba(39,209,127,0.12)' }}
          />
          {/* Middle dashed ring */}
          <div
            ref={ring2Ref}
            className="absolute rounded-full opacity-0"
            style={{ inset: '18px', border: '1.5px dashed rgba(39,209,127,0.22)' }}
          />
          {/* Inner solid ring */}
          <div
            ref={ring1Ref}
            className="absolute rounded-full opacity-0"
            style={{ inset: '36px', border: '2px solid rgba(39,209,127,0.38)' }}
          />
          {/* Eye image */}
          <div
            ref={eyeRef}
            className="absolute rounded-full overflow-hidden opacity-0"
            style={{
              inset: '52px',
              boxShadow: '0 0 0 1.5px rgba(39,209,127,0.40), 0 0 55px rgba(39,209,127,0.18), 0 0 110px rgba(39,209,127,0.07)',
            }}
          >
            <img
              src="/analysis_retina_closeup.jpg"
              alt="Retinal scan"
              className="w-full h-full object-cover"
            />
            {/* Mint tint */}
            <div
              className="absolute inset-0"
              style={{
                background: 'radial-gradient(circle at 55% 48%, rgba(39,209,127,0.26) 0%, rgba(39,209,127,0.07) 50%, transparent 72%)',
                mixBlendMode: 'screen',
              }}
            />
            {/* Scan line */}
            <div
              ref={scanRef}
              className="absolute left-0 right-0 opacity-0 pointer-events-none"
              style={{
                height: '2px',
                background: 'linear-gradient(90deg, transparent 0%, rgba(39,209,127,0.7) 30%, rgba(39,209,127,1) 50%, rgba(39,209,127,0.7) 70%, transparent 100%)',
                boxShadow: '0 0 10px rgba(39,209,127,0.65)',
              }}
            />
          </div>

          {/* Anomaly dots */}
          <div ref={dotsRef} className="absolute inset-0 pointer-events-none">
            {[
              { top: '39%', left: '59%', r: 9 },
              { top: '55%', left: '54%', r: 6 },
              { top: '45%', left: '47%', r: 5 },
            ].map((d, i) => (
              <div
                key={i}
                className="absolute rounded-full bg-mint animate-pulse"
                style={{
                  top: d.top, left: d.left,
                  width: d.r, height: d.r,
                  opacity: 0,
                  boxShadow: '0 0 8px rgba(39,209,127,0.85)',
                  animationDelay: `${i * 0.4}s`,
                }}
              />
            ))}
          </div>
        </div>

        {/* RIGHT: Stat cards ───────────────────────────────────────────────── */}
        <div ref={statsRef} className="flex flex-col gap-4 w-[210px] flex-none">
          {stats.map((s, i) => (
            <div
              key={i}
              className="rounded-2xl p-5 flex flex-col gap-2"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.07)',
                backdropFilter: 'blur(16px)',
              }}
            >
              <div className="flex items-center gap-2 text-white/35">
                <s.icon className="w-3.5 h-3.5 flex-none" />
                <span className="text-[11px] tracking-wide">{s.label}</span>
              </div>
              <div className="text-[1.55rem] font-bold text-white leading-none">{s.value}</div>
              <div className="text-[11px] text-white/30">{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
