import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Button } from '@/components/ui/button';
import { ArrowRight, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

gsap.registerPlugin(ScrollTrigger);

export default function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const blobRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const section = sectionRef.current;
    const blob = blobRef.current;
    const card = cardRef.current;
    const image = imageRef.current;
    const text = textRef.current;

    if (!section || !blob || !card || !image || !text) return;

    const ctx = gsap.context(() => {
      // Faster load animation (auto-play on mount)
      const loadTl = gsap.timeline({ defaults: { ease: 'power2.out' } });

      loadTl
        .fromTo(blob, { opacity: 0, scale: 0.96 }, { opacity: 1, scale: 1, duration: 0.5 })
        .fromTo(card, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6 }, 0.05)
        .fromTo(image, { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.6 }, 0.1)
        .fromTo(
          text.children,
          { opacity: 0, y: 16 },
          { opacity: 1, y: 0, duration: 0.4, stagger: 0.06 },
          0.15
        );

      // Faster scroll-driven exit animation
      const scrollTl = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: 'top top',
          end: '+=100%',
          pin: true,
          scrub: 0.4,
          onLeaveBack: () => {
            gsap.set([blob, card, image, text.children], {
              opacity: 1,
              x: 0,
              y: 0,
              scale: 1,
            });
          },
        },
      });

      // EXIT (60-100%): Faster exit
      scrollTl
        .fromTo(card, { x: 0, opacity: 1 }, { x: '-12vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(image, { y: 0, scale: 1 }, { y: '6vh', scale: 0.98, ease: 'power2.in' }, 0.6)
        .fromTo(text.children, { x: 0, opacity: 1 }, { x: '6vw', opacity: 0.3, ease: 'power2.in' }, 0.6)
        .fromTo(blob, { scale: 1, opacity: 0.1 }, { scale: 1.05, opacity: 0.04, ease: 'power2.in' }, 0.6);
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-screen overflow-hidden bg-offwhite flex items-center justify-center z-10"
    >
      {/* Background Blob */}
      <div
        ref={blobRef}
        className="absolute blob-mint rounded-[44px]"
        style={{
          left: '18vw',
          top: '10vh',
          width: '64vw',
          height: '80vh',
        }}
      />

      {/* Main Card */}
      <div
        ref={cardRef}
        className="absolute bg-offwhite rounded-[28px] card-shadow overflow-hidden"
        style={{
          left: '10vw',
          top: '14vh',
          width: '80vw',
          height: '72vh',
        }}
      >
        {/* Dot Pattern Background */}
        <div className="absolute inset-0 dot-pattern opacity-50" />

        {/* Hero Image */}
        <div
          ref={imageRef}
          className="absolute rounded-[22px] overflow-hidden"
          style={{
            left: '3vw',
            top: '6vh',
            width: '34vw',
            height: '60vh',
          }}
        >
          <img
            src="/hero_patient_exam.jpg"
            alt="Patient undergoing eye exam"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-navy/20 to-transparent" />
        </div>

        {/* Text Content */}
        <div
          ref={textRef}
          className="absolute flex flex-col"
          style={{
            left: '42vw',
            top: '18vh',
            width: '34vw',
          }}
        >
          {/* Eyebrow */}
          <div className="flex items-center gap-2 mb-6">
            <Sparkles className="w-4 h-4 text-mint" />
            <span className="text-xs font-semibold tracking-[0.15em] uppercase text-mint">
              AI-Powered Retinal Screening
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl lg:text-5xl xl:text-[56px] font-bold text-navy leading-[1.1] mb-6">
            AI Eye Disease Detection
          </h1>

          {/* Body */}
          <p className="text-base lg:text-lg text-navy/60 leading-relaxed mb-8 max-w-md">
            Detect Diabetic Retinopathy, Glaucoma, Pathological Myopia, and Cataract from retinal images using advanced AI.
          </p>

          {/* CTA Row - Simplified for hackathon */}
          <div className="flex items-center gap-4 mt-auto">
            <Button
              size="lg"
              className="bg-navy hover:bg-navy/90 text-white px-8 rounded-full group"
              onClick={() => navigate('/app')}
            >
              Get Started
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>
        </div>

        {/* Decorative Icon Circle */}
        <div
          className="absolute w-14 h-14 rounded-full bg-mint/15 flex items-center justify-center"
          style={{ right: '4vw', top: '10vh' }}
        >
          <div className="w-3 h-3 rounded-full bg-mint animate-pulse" />
        </div>
      </div>
    </section>
  );
}
