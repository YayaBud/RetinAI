import Navigation from '../components/Navigation';
import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  Microscope,
  Brain,
  Shield,
  Share2,
  Focus,
  SlidersHorizontal,
  FileCheck,
  Rocket,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

gsap.registerPlugin(ScrollTrigger);

/* ── Color palette matching the Stitch teal-blue design ─────────────── */
const C = {
  bg: '#3b6a7a',   // main page teal-blue
  bgDark: '#2f5968',   // slightly darker teal for nav/sections
  bgDeep: '#264b59',   // deepest teal (cards)
  bgCard: '#2a5160',   // feature card bg
  bgOverlay: '#1e4250',   // overlays / grid bg
  accent: '#27D17F',   // mint green accent
  text: '#f0f5fa',   // primary text
  textSub: 'rgba(255,255,255,0.65)', // body text
  textMuted: 'rgba(255,255,255,0.35)', // labels
};

/* ── ASCII Eye Frames ─────────────────────────────────────────────────── */
const f0 = [
  '                                        ⡄                                        ',
  '                                        ⡇  ⠈                                     ',
  '                                        ⡇⡀                                       ',
  '                              ⠠ ⠀  ⠀⠁ ⠀⢰⠇ ⠀⠠                                   ',
  '                          ⠠ ⠀ ⠀⠀⢀ ⠀⠇⢀⡇⢸⣸ ⠀⠀  ⠁                                ',
  '                      ⠈ ⡀⠰ ⠀⠀⠀ ⠄ ⢸⢸⢠ ⣾⢸⢸⣿⢸⢀⢠ ⡆⡇ ⠐                           ',
  '                    ⠁ ⠀ ⢃ ⢀ ⠀⢀ ⢢⡀ ⠆⣇⢿⢸⣾⣿⣇⣿⣿⣼⣿⣸⢷⡇⣼⢀ ⣠⠊ ⣠⠆              ',
  '               ⠈⠢  ⡑⢄ ⠁ ⠑⢄⠙⢦⡀⠢⠙⡦⣈⢧⡻⣜⠼⣜⢯⣿⣿⣿⣿⣿⣿⣿⣿⣼⣹⢣⢣⢡⠞⣁⣴⠞⡁   ⡠  ⠤  ⡠   ',
  '                ⠑⠠⡈⠒⠥⣀ ⠐⠄⡉⠢⣝⡲⢬⡪⣎⢧⠽⠟⡺⠿⠛⠋⠉⠉⠉⠙⠛⠛⠿⣟⡻⢷⣾⣫⠥⡺⠕⣀⠤⡊ ⢠ ⢀⡠⠂⢀⡠⠊    ',
  '             ⠒⠤⣀⠑⠢⠬⣽⣒⠤⠈⠒⡦⢭⣟⠚⣩⠰⠊⠁   ⢀⡀⡀       ⢀ ⠉⠓⢮⣝⡳⢻⣭⠖⣋⠠⣀⡴⠞⡩⠄⠚⠁  ⠄      ',
  '           ⠈ ⠉⠒⠀⠬⠍⠛⠛⣚⣩⡆⠋⠁⣀⣴⣶⠏⣠⡞⣡⣶⣶⣶⡄    ⠻⣷⣦⣀⠈⠛⢶⣬⣓⣒⢛⣃⣉⠠⠔ ⠠⠂⠁   ⠠       ',
  '         ⠈⠁⠐⠢⠤⣁⣒⣒⣛⣂⣶⡟⠟⠉⢀⣤⣾⣿⣿⡏⢠⢶⡃⢿⣿⣿⠿⠁    ⢹⣿⣿⣷⣤ ⠈⠻⢯⣟⣂⣂⣒⣒⣒⣈⡩⠥⠐⠈⠁  ⠠ ⠈⠉⠁   ',
  '        ⠈⠉⠉⠉ ⠐⠒⣒⣛⣿⣿⣛⠉  ⠠⣾⣿⣿⣿⣿⡅⢊⠎⣹ ⠉⠁       ⢸⣿⣿⣿⣿⣷  ⠉⣛⠒⢲⠆⠡⠤⠤⠤⠒⠒ ⠈   ⢀⡀     ',
  '      ⠈ ⡀⠠⠤⠐⠒⠒⣒⠒⠚⠳⠼⠛⠿⣶⣥⡠⡀⠙⢿⣿⣿⣿⣇ ⠘⠄⠃        ⣮⣿⣿⣿⠟⠃ ⢀⣴⣶⠿⠛⢿⣽⣛⠋⣉⣉⠉⠒⠒⠒⠂⠐ ⠉        ',
  '       ⠤⠄ ⠤⠐ ⠈⢉⡠⠄⣀⠤⠒⣈⡭⠾⢙⡿⣾⣤⣂ ⣉⠑        ⠐⠊⣉⠠⣀⣬⡶⢿⣟⠯⢍⡛⠶⡤⠉⠑⠢⢄  ⠉ ⠂⠠           ',
  '        ⠠  ⠀ ⠒⡡⠔⠈ ⠢⠋⠁⠂ ⣡⠴⢃⣵⢟⡟⣷⣾⣿⣶⣶⣤⣤⣤⣴⣶⣦⣬⡷⣶⢿⢯⡳⣌⠢⢍⠛⠦⠌⠑⠠  ⠲⠤⡉⠢ ⠈ ⡀        ',
  '        ⠂   ⠄  ⠁⠘⢁⢀⠔⠁⠁⣽⢣⣇⡏⡏⣿⡟⣿⣿⢿⣿⣿⢸⡵⢹⣯⠆⠑⢜⢣⡀⠉⠢⣈⠂   ⠂⡀  ⠑⢄            ',
  '            ⠔  ⡠⠈  ⣰⠑⢸⣹⢹⢿⣿⡇⣿⣿⢸⡟⢸⠈⣷⠁⠙⢇  ⠙⢦⡀⠈⠃⢄  ⠐   ⠄              ',
  '             ⡀⠁  ⠀⠁⠃⠇⢸⢸⢸⠸⡏⡇⢸⣿⢸⣧⢨ ⡝⡏ ⠈⠂  ⢀  ⡀⠑⡀  ⠈                  ',
  '               ⠒     ⡸⢸⢸ ⣧⢿⢸⣿ ⣿⠈ ⠇⡇  ⠐   ⠈⠄   ⠄                   ',
  '                ⠈ ⢀ ⢁⠘ ⡀⠸⡌⢸⣿ ⡏ ⢀ ⡄ ⣤  ⠐      ⢠                     ',
  '                 ⠐   ⢸ ⡀⠇⠸⠃⢸⣿ ⠇ ⢰      ⠈ ⠂                            ',
  '                    ⡀ ⢀  ⠁ ⠂⠸⡟   ⠂                ⠠                       ',
  '                  ⠁      ⠂  ⡇   ⠀                                          ',
  '                      ⠄    ⡇⠤                                               ',
  '                           ⡇  ⠈⠁                                             ',
  '                          ⢰                                                   ',
];

const f1 = [
  '                                        ⡄                                        ',
  '                                        ⡇  ⠈                                     ',
  '                                        ⡇⡀                                       ',
  '                              ⠠ ⠀  ⠀⠁ ⠀⢰⠇ ⠀⠠                                   ',
  '                          ⠠ ⠀ ⠀⠀⢀ ⠀⠇⢀⡇⢸⣸ ⠀⠀  ⠁                                ',
  '                      ⠈ ⡀⠰ ⠀⠀⠀ ⠄ ⢸⢸⢠ ⣾⢸⢸⣿⢸⢀⢠ ⡆⡇ ⠐                           ',
  '                    ⠁ ⠀ ⢃ ⢀ ⠀⢀ ⢢⡀ ⠆⣇⢿⢸⣾⣿⣇⣿⣿⣼⣿⣸⢷⡇⣼⢀ ⣠⠊ ⣠⠆              ',
  '               ⠈⠢  ⡑⢄ ⠁ ⠑⢄⠙⢦⡀⠢⠙⡦⣈⢧⡻⣜⠼⣜⢯⣿⣿⣿⣿⣿⣿⣿⣿⣼⣹⢣⢣⢡⠞⣁⣴⠞⡁   ⡠  ⠤  ⡠   ',
  '                ⠑⠠⡈⠒⠥⣀ ⠐⠄⡉⠢⣝⡲⢬⡪⣎⢧⠽⠟⡺⠿⠛⠋⠉⠉⠉⠙⠛⠛⠿⣟⡻⢷⣾⣫⠥⡺⠕⣀⠤⡊ ⢠ ⢀⡠⠂⢀⡠⠊    ',
  '             ⠒⠤⣀⠑⠢⠬⣽⣒⠤⠈⠒⡦⢭⣟⠚⣩⠰⠊⠁   ⢀⡀⡀       ⢀ ⠉⠓⢮⣝⡳⢻⣭⠖⣋⠠⣀⡴⠞⡩⠄⠚⠁  ⠄      ',
  '           ⠈ ⠉⠒⠀⠬⠍⠛⠛⣚⣩⡆⠋⠁              ⠻⣷⣦⣀⠈⠛⢶⣬⣓⣒⢛⣃⣉⠠⠔ ⠠⠂⠁   ⠠       ',
  '         ⠈⠁⠐⠢⠤⣁⣒⣒⣛⣂⣶⡟⠟⠉⢀⣤⣴⣶⠏⣠⡞⣡⣶⣶⣶⡄   ⢹⣿⣿⣷⣤ ⠈⠻⢯⣟⣂⣂⣒⣒⣒⣈⡩⠥⠐⠈⠁  ⠠ ⠈⠉⠁   ',
  '        ⠈⠉⠉⠉ ⠐⠒⣒⣛⣿⣿⣛⠉  ⠠⣾⣿⣿⡏⢠⢶⡃⢿⣿⣿⠿⠁       ⢸⣿⣿⣿⣿⣷  ⠉⣛⠒⢲⠆⠡⠤⠤⠤⠒⠒ ⠈   ⢀⡀     ',
  '      ⠈ ⡀⠠⠤⠐⠒⠒⣒⠒⠚⠳⠼⠛⠿⣶⣥⡠⡀⣿⣿⡅⢊⠎⣹ ⠉⠁       ⣮⣿⣿⣿⠟⠃ ⢀⣴⣶⠿⠛⢿⣽⣛⠋⣉⣉⠉⠒⠒⠒⠂⠐ ⠉        ',
  '       ⠤⠄ ⠤⠐ ⠈⢉⡠⠄⣀⠤⠒⣈⡭⠾⢙⡿⣾⢿⣿⣿⣇ ⠘⠄⠃               ⣀⣬⡶⢿⣟⠯⢍⡛⠶⡤⠉⠑⠢⢄  ⠉ ⠂⠠           ',
  '        ⠠  ⠀ ⠒⡡⠔⠈ ⠢⠋⠁⠂ ⣡⠴⢃⣵⢟⡟⣷⣤⣂ ⣉⠑     ⣴⣶⣦⣬⡷⣶⢿⢯⡳⣌⠢⢍⠛⠦⠌⠑⠠  ⠲⠤⡉⠢ ⠈ ⡀        ',
  '        ⠂   ⠄  ⠁⠘⢁⢀⠔⠁⠁⣽⢣⣇⡏⡏⣿⡟⣿⣿⢿⣿⣿⢸⡵⢹⣯⠆⠑⢜⢣⡀⠉⠢⣈⠂   ⠂⡀  ⠑⢄            ',
  '            ⠔  ⡠⠈  ⣰⠑⢸⣹⢹⢿⣿⡇⣿⣿⢸⡟⢸⠈⣷⠁⠙⢇  ⠙⢦⡀⠈⠃⢄  ⠐   ⠄              ',
  '             ⡀⠁  ⠀⠁⠃⠇⢸⢸⢸⠸⡏⡇⢸⣿⢸⣧⢨ ⡝⡏ ⠈⠂  ⢀  ⡀⠑⡀  ⠈                  ',
  '               ⠒     ⡸⢸⢸ ⣧⢿⢸⣿ ⣿⠈ ⠇⡇  ⠐   ⠈⠄   ⠄                   ',
  '                ⠈ ⢀ ⢁⠘ ⡀⠸⡌⢸⣿ ⡏ ⢀ ⡄ ⣤  ⠐      ⢠                     ',
  '                 ⠐   ⢸ ⡀⠇⠸⠃⢸⣿ ⠇ ⢰      ⠈ ⠂                            ',
  '                    ⡀ ⢀  ⠁ ⠂⠸⡟   ⠂                ⠠                       ',
  '                  ⠁      ⠂  ⡇   ⠀                                          ',
  '                      ⠄    ⡇⠤                                               ',
  '                           ⡇  ⠈⠁                                             ',
  '                          ⢰                                                   ',
];

const f2 = [
  '                                        ⡄                                        ',
  '                                        ⡇  ⠈                                     ',
  '                                        ⡇⡀                                       ',
  '                              ⠠ ⠀  ⠀⠁ ⠀⢰⠇ ⠀⠠                                   ',
  '                          ⠠ ⠀ ⠀⠀⢀ ⠀⠇⢀⡇⢸⣸ ⠀⠀  ⠁                                ',
  '                      ⠈ ⡀⠰ ⠀⠀⠀ ⠄ ⢸⢸⢠ ⣾⢸⢸⣿⢸⢀⢠ ⡆⡇ ⠐                           ',
  '                    ⠁ ⠀ ⢃ ⢀ ⠀⢀ ⢢⡀ ⠆⣇⢿⢸⣾⣿⣇⣿⣿⣼⣿⣸⢷⡇⣼⢀ ⣠⠊ ⣠⠆              ',
  '               ⠈⠢  ⡑⢄ ⠁ ⠑⢄⠙⢦⡀⠢⠙⡦⣈⢧⡻⣜⠼⣜⢯⣿⣿⣿⣿⣿⣿⣿⣿⣼⣹⢣⢣⢡⠞⣁⣴⠞⡁   ⡠  ⠤  ⡠   ',
  '                ⠑⠠⡈⠒⠥⣀ ⠐⠄⡉⠢⣝⡲⢬⡪⣎⢧⠽⠟⡺⠿⠛⠋⠉⠉⠉⠙⠛⠛⠿⣟⡻⢷⣾⣫⠥⡺⠕⣀⠤⡊ ⢠ ⢀⡠⠂⢀⡠⠊    ',
  '             ⠒⠤⣀⠑⠢⠬⣽⣒⠤⠈⠒⡦⢭⣟⠚⣩⠰⠊⠁                                 ⡴⠞⡩⠄⠚⠁  ⠄      ',
  '           ⠈ ⠉⠒⠀⠬⠍⠛⠛⣚⣩⡆⠋⠁              ⠉⠓⢮⣝⡳⢻⣭⠖⣋⠠⣀⠈⠛⢶⣬⣓⣒⢛⣃⣉⠠⠔ ⠠⠂⠁   ⠠       ',
  '         ⠈⠁⠐⠢⠤⣁⣒⣒⣛⣂⣶⡟⠟⠉⢀⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⠈⠻⢯⣟⣂⣂⣒⣒⣒⣈⡩⠥⠐⠈⠁  ⠠ ⠈⠉⠁   ',
  '        ⠈⠉⠉⠉ ⠐⠒⣒⣛⣿⣿⣛⠉  ⠠                         ⠉⣛⠒⢲⠆⠡⠤⠤⠤⠒⠒ ⠈   ⢀⡀     ',
  '      ⠈ ⡀⠠⠤⠐⠒⠒⣒⠒⠚⠳⠼⠛⠿⣶⣥⡠⡀                                   ⢀⣴⣶⠿⠛⢿⣽⣛⠋⣉⣉⠉⠒⠒⠒⠂⠐ ⠉        ',
  '       ⠤⠄ ⠤⠐ ⠈⢉⡠⠄⣀⠤⠒⣈⡭⠾⢙⡿⣾⣤⣂                              ⣀⣬⡶⢿⣟⠯⢍⡛⠶⡤⠉⠑⠢⢄  ⠉ ⠂⠠           ',
  '        ⠠  ⠀ ⠒⡡⠔⠈ ⠢⠋⠁⠂ ⣡⠴⢃⣵⢟⡟⣷                 ⣴⣶⣦⣬⡷⣶⢿⢯⡳⣌⠢⢍⠛⠦⠌⠑⠠  ⠲⠤⡉⠢ ⠈ ⡀        ',
  '        ⠂   ⠄  ⠁⠘⢁⢀⠔⠁⠁⣽⢣⣇⡏⡏⣿⡟⣿⣿⢿⣿⣿⢸⡵⢹⣯⠆⠑⢜⢣⡀⠉⠢⣈⠂   ⠂⡀  ⠑⢄            ',
  '            ⠔  ⡠⠈  ⣰⠑⢸⣹⢹⢿⣿⡇⣿⣿⢸⡟⢸⠈⣷⠁⠙⢇  ⠙⢦⡀⠈⠃⢄  ⠐   ⠄              ',
  '             ⡀⠁  ⠀⠁⠃⠇⢸⢸⢸⠸⡏⡇⢸⣿⢸⣧⢨ ⡝⡏ ⠈⠂  ⢀  ⡀⠑⡀  ⠈                  ',
  '               ⠒     ⡸⢸⢸ ⣧⢿⢸⣿ ⣿⠈ ⠇⡇  ⠐   ⠈⠄   ⠄                   ',
  '                ⠈ ⢀ ⢁⠘ ⡀⠸⡌⢸⣿ ⡏ ⢀ ⡄ ⣤  ⠐      ⢠                     ',
  '                 ⠐   ⢸ ⡀⠇⠸⠃⢸⣿ ⠇ ⢰      ⠈ ⠂                            ',
  '                    ⡀ ⢀  ⠁ ⠂⠸⡟   ⠂                ⠠                       ',
  '                  ⠁      ⠂  ⡇   ⠀                                          ',
  '                      ⠄    ⡇⠤                                               ',
  '                           ⡇  ⠈⠁                                             ',
  '                          ⢰                                                   ',
];

/* ── Audit log mock data ───────────────────────────────────────────── */
const auditLog = [
  { time: '14:22:04', action: 'SCAN COMPLETED', status: 'VERIFIED', hash: '0x7f3a...9b2c' },
  { time: '14:18:31', action: 'AI OVERLAY GENERATED', status: 'VERIFIED', hash: '0x3e8d...4a1f' },
  { time: '14:15:12', action: 'IMAGE UPLOADED', status: 'VERIFIED', hash: '0x9c2b...7e5d' },
  { time: '14:12:47', action: 'SESSION INITIATED', status: 'VERIFIED', hash: '0xa1f4...3c8e' },
];

export default function LandingPage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const archRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero entrance
      if (heroRef.current) {
        const h = heroRef.current;
        gsap.fromTo(h.querySelectorAll('.hero-anim'),
          { opacity: 0, y: 50 },
          { opacity: 1, y: 0, stagger: 0.12, duration: 0.9, ease: 'power3.out', delay: 0.2 }
        );
        gsap.fromTo(h.querySelector('.ascii-eye'),
          { opacity: 0, scale: 0.85 },
          { opacity: 1, scale: 1, duration: 1.4, ease: 'power2.out', delay: 0.6 }
        );
      }

      // Stats entrance
      if (statsRef.current) {
        gsap.fromTo(statsRef.current.children,
          { opacity: 0, y: 30, scale: 0.95 },
          {
            opacity: 1, y: 0, scale: 1, stagger: 0.1, duration: 0.7, ease: 'back.out(1.4)',
            scrollTrigger: { trigger: statsRef.current, start: 'top 85%' },
          }
        );
      }

      // Features entrance
      if (featuresRef.current) {
        gsap.fromTo(featuresRef.current.querySelectorAll('.feature-card'),
          { opacity: 0, y: 40 },
          {
            opacity: 1, y: 0, stagger: 0.1, duration: 0.7, ease: 'power2.out',
            scrollTrigger: { trigger: featuresRef.current, start: 'top 80%' },
          }
        );
      }

      // Architecture section
      if (archRef.current) {
        gsap.fromTo(archRef.current.querySelectorAll('.arch-anim'),
          { opacity: 0, y: 30 },
          {
            opacity: 1, y: 0, stagger: 0.08, duration: 0.6, ease: 'power2.out',
            scrollTrigger: { trigger: archRef.current, start: 'top 80%' },
          }
        );
      }

      // CTA entrance
      if (ctaRef.current) {
        gsap.fromTo(ctaRef.current,
          { opacity: 0, y: 40 },
          {
            opacity: 1, y: 0, duration: 0.8, ease: 'power2.out',
            scrollTrigger: { trigger: ctaRef.current, start: 'top 85%' },
          }
        );
      }
    });

    return () => ctx.revert();
  }, []);

  const SG = { fontFamily: "'Space Grotesk', sans-serif" };

  return (
    <div className="relative min-h-screen overflow-x-hidden" style={{ background: C.bg }}>
      <Navigation />

      {/* ═══════════════════════════════════════════════════════════════════
          HERO SECTION
      ═══════════════════════════════════════════════════════════════════ */}
      <section
        id="accuracy"
        className="relative min-h-screen flex items-center pt-20"
        style={{ background: `linear-gradient(180deg, ${C.bgDark} 0%, ${C.bg} 100%)` }}
      >
        {/* Large radial glow */}
        <div
          className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] rounded-full pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(39,209,127,0.06) 0%, transparent 55%)' }}
        />

        <div ref={heroRef} className="relative z-10 w-full max-w-[1400px] mx-auto px-6 lg:px-12 flex flex-col lg:flex-row items-center gap-12 lg:gap-0">
          {/* Left text */}
          <div className="flex-1 min-w-0">
            {/* Badge */}
            <div
              className="hero-anim inline-flex items-center gap-2.5 px-5 py-2.5 rounded-lg mb-8"
              style={{ background: 'rgba(39,209,127,0.08)', border: '1px solid rgba(39,209,127,0.18)' }}
            >
              <span className="w-2 h-2 rounded-full bg-[#27D17F] animate-pulse flex-none" />
              <span className="text-[11px] font-medium tracking-[0.18em] uppercase" style={{ color: C.text }}>
                Clinical Grade Diagnostics
              </span>
            </div>

            {/* Headline */}
            <h1
              className="hero-anim text-[clamp(2.8rem,5.5vw,5.5rem)] font-bold leading-[1.05] mb-6"
              style={{ ...SG, color: C.text }}
            >
              Surgical<br />accuracy<br />
              <span className="italic" style={{ color: C.accent }}>
                at the speed of<br />light.
              </span>
            </h1>

            {/* Body */}
            <p className="hero-anim text-base lg:text-lg leading-relaxed max-w-[32rem] mb-8" style={{ color: C.textSub }}>
              Retina AI leverages specialized optical neural networks to identify
              pathologies with clinical precision. Experience the next
              evolution of diagnostic intelligence.
            </p>

            {/* CTAs */}
            <div className="hero-anim flex items-center gap-4">
              <Button
                size="lg"
                className="bg-[#27D17F] hover:bg-[#22b86e] text-[#0a1e2c] font-bold px-8 rounded-xl group shadow-lg shadow-[#27D17F]/20"
                onClick={() => navigate('/app')}
              >
                Launch Observatory
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button
                size="lg"
                variant="ghost"
                className="rounded-xl font-medium border border-white/10 hover:bg-white/5"
                style={{ color: 'rgba(255,255,255,0.7)' }}
                onClick={() => navigate('/login')}
              >
                Sign In
              </Button>
            </div>
          </div>

          {/* Right: ASCII Eye Component (Animated) */}
          <div className="flex-1 flex justify-center lg:justify-end relative items-center">
            {/* Ambient continuous glow behind eye */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3/4 h-1/2 blur-[80px] pointer-events-none rounded-full" style={{ background: 'rgba(39,209,127,0.18)' }} />
            
            <div className="ascii-eye relative z-10 flex items-center justify-center">
              <AsciiEye />
            </div>
          </div>
        </div>

        {/* Bottom gradient to blend into stats */}
        <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none" style={{ background: `linear-gradient(transparent, ${C.bg})` }} />
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          STATS BAR
      ═══════════════════════════════════════════════════════════════════ */}
      <section className="relative py-14" style={{ background: C.bg }}>
        <div
          ref={statsRef}
          className="max-w-[1400px] mx-auto px-6 lg:px-12 flex justify-center"
        >
          <div
            className="rounded-2xl py-12 px-20 text-center w-full max-w-lg relative overflow-hidden"
            style={{ background: C.bgCard, boxShadow: '0 8px 32px rgba(0,0,0,0.15)' }}
          >
            {/* Subtle accent glow inside card */}
            <div
              className="absolute top-0 left-1/2 -translate-x-1/2 w-48 h-1 rounded-b-full"
              style={{ background: 'linear-gradient(90deg, transparent, rgba(39,209,127,0.4), transparent)' }}
            />
            <div className="text-5xl lg:text-7xl font-bold mb-3" style={{ ...SG, color: C.accent }}>
              98.6%
            </div>
            <div className="text-[11px] font-medium tracking-[0.25em] uppercase" style={{ color: C.textMuted }}>
              Meta Routing Accuracy
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          THE OBSERVATORY SUITE (Features)
      ═══════════════════════════════════════════════════════════════════ */}
      <section
        id="features"
        className="relative py-24 lg:py-32"
        style={{ background: C.bg }}
      >
        {/* Section separator gradient */}
        <div className="absolute top-0 left-0 right-0 h-32 pointer-events-none" style={{ background: `linear-gradient(${C.bg}, transparent)` }} />

        <div ref={featuresRef} className="max-w-[1400px] mx-auto px-6 lg:px-12">
          {/* Section header */}
          <div className="feature-card flex flex-col lg:flex-row items-start lg:items-end justify-between gap-6 mb-16">
            <div>
              <span className="text-[11px] font-medium tracking-[0.2em] uppercase mb-4 block" style={{ color: C.accent }}>
                Core Ecosystem
              </span>
              <h2 className="text-4xl lg:text-5xl font-bold" style={{ ...SG, color: C.text }}>
                The Observatory Suite.
              </h2>
            </div>
            <p
              className="text-base max-w-md lg:text-right leading-relaxed"
              style={{ color: C.textSub, borderLeft: `2px solid rgba(39,209,127,0.2)`, paddingLeft: '1rem' }}
            >
              An end-to-end diagnostic environment designed for
              clinical speed and absolute data integrity.
            </p>
          </div>

          {/* Feature Cards — 2×2 */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
            {/* Card 1 — Big (3 cols) */}
            <div
              className="feature-card lg:col-span-3 rounded-2xl p-8 lg:p-10 flex flex-col justify-between min-h-[320px] group transition-all duration-500 hover:-translate-y-2"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)' }}
            >
              <div>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6" style={{ background: 'rgba(39,209,127,0.1)' }}>
                  <Microscope className="w-6 h-6" style={{ color: C.accent }} />
                </div>
                <h3 className="text-2xl font-bold mb-3" style={{ ...SG, color: C.text }}>
                  The Diagnostic Hub
                </h3>
                <p className="text-base leading-relaxed max-w-lg" style={{ color: C.textSub }}>
                  Centralized processing for high-volume imaging. Instantly route
                  results to relevant departments with zero-latency synchronization.
                </p>
              </div>
              <a
                href="#"
                className="inline-flex items-center gap-2 text-[11px] font-medium tracking-[0.15em] uppercase mt-6 hover:gap-3 transition-all"
                style={{ color: C.accent }}
              >
                EXPLORE <ArrowRight className="w-3.5 h-3.5" />
              </a>
            </div>

            {/* Card 2 — Small (2 cols) */}
            <div
              className="feature-card lg:col-span-2 rounded-2xl p-8 lg:p-10 flex flex-col justify-between min-h-[320px] group transition-all duration-500 hover:-translate-y-2"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)' }}
            >
              <div>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6" style={{ background: 'rgba(39,209,127,0.1)' }}>
                  <Brain className="w-6 h-6" style={{ color: C.accent }} />
                </div>
                <h3 className="text-2xl font-bold mb-3" style={{ ...SG, color: C.text }}>
                  Neural Analytics
                </h3>
                <p className="text-base leading-relaxed" style={{ color: C.textSub }}>
                  Deep learning models trained on 15 years of clinical data provide
                  secondary verification layers that exceed human capability.
                </p>
              </div>
              <a
                href="#"
                className="inline-flex items-center gap-2 text-[11px] font-medium tracking-[0.15em] uppercase mt-6 hover:gap-3 transition-all"
                style={{ color: C.accent }}
              >
                EXPLORE AI MODELS <ArrowRight className="w-3.5 h-3.5" />
              </a>
            </div>

            {/* Card 3 — Small (2 cols) */}
            <div
              className="feature-card lg:col-span-2 rounded-2xl p-8 lg:p-10 flex flex-col justify-between min-h-[320px] group transition-all duration-500 hover:-translate-y-2"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)' }}
            >
              <div>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6" style={{ background: 'rgba(39,209,127,0.1)' }}>
                  <Shield className="w-6 h-6" style={{ color: C.accent }} />
                </div>
                <h3 className="text-2xl font-bold mb-3" style={{ ...SG, color: C.text }}>
                  HIPAA Obsidian
                </h3>
                <p className="text-base leading-relaxed" style={{ color: C.textSub }}>
                  End-to-end encrypted medical data storage. Decentralized architecture
                  ensures zero-point failure and absolute privacy.
                </p>
              </div>
              {/* Mini bar chart visual */}
              <div className="flex items-end gap-1.5 mt-6">
                {[40, 55, 70, 85].map((h, i) => (
                  <div
                    key={i}
                    className="w-2.5 rounded-sm"
                    style={{ height: `${h * 0.4}px`, background: `rgba(39,209,127,${0.35 + i * 0.12})` }}
                  />
                ))}
              </div>
            </div>

            {/* Card 4 — Big (3 cols) */}
            <div
              className="feature-card lg:col-span-3 rounded-2xl p-8 lg:p-10 flex flex-col justify-between min-h-[320px] relative overflow-hidden group transition-all duration-500 hover:-translate-y-2"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05)' }}
            >
              <div>
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6" style={{ background: 'rgba(39,209,127,0.1)' }}>
                  <Share2 className="w-6 h-6" style={{ color: C.accent }} />
                </div>
                <h3 className="text-2xl font-bold mb-3" style={{ ...SG, color: C.text }}>
                  Peer-to-Peer Relay
                </h3>
                <p className="text-base leading-relaxed max-w-lg" style={{ color: C.textSub }}>
                  Instant consultation rooms. Share DICOM files and AI overlays with
                  specialists across the globe in a unified virtual observatory.
                </p>
              </div>

              {/* Decorative network nodes */}
              <div className="absolute top-6 right-6 opacity-60">
                <svg width="100" height="80" viewBox="0 0 100 80" fill="none">
                  <circle cx="20" cy="20" r="6" stroke={C.accent} strokeWidth="1.5" fill="none" />
                  <circle cx="80" cy="20" r="6" stroke={C.accent} strokeWidth="1.5" fill="none" />
                  <circle cx="50" cy="60" r="6" stroke={C.accent} strokeWidth="1.5" fill="none" />
                  <circle cx="50" cy="20" r="3" fill={C.accent} />
                  <line x1="26" y1="20" x2="44" y2="20" stroke={C.accent} strokeWidth="1" />
                  <line x1="56" y1="20" x2="74" y2="20" stroke={C.accent} strokeWidth="1" />
                  <line x1="47" y1="23" x2="23" y2="57" stroke={C.accent} strokeWidth="1" opacity="0.4" />
                  <line x1="53" y1="23" x2="77" y2="57" stroke={C.accent} strokeWidth="1" opacity="0.4" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          PRECISION METRICS / SCANNING ARCHITECTURE
      ═══════════════════════════════════════════════════════════════════ */}
      <section
        id="architecture"
        className="relative py-24 lg:py-32"
        style={{ background: C.bg }}
      >
        <div ref={archRef} className="max-w-[1400px] mx-auto px-6 lg:px-12">
          {/* Header */}
          <div className="arch-anim flex flex-col lg:flex-row items-start gap-10 mb-20">
            <div className="flex-1">
              <span className="text-[11px] font-medium tracking-[0.2em] uppercase mb-4 block" style={{ color: C.accent }}>
                Precision Metrics
              </span>
              <h2 className="text-4xl lg:text-5xl font-bold leading-tight" style={{ ...SG, color: C.text }}>
                Advanced Scanning<br />Architecture
              </h2>
            </div>
            {/* Performance card */}
            <div
              className="arch-anim rounded-2xl p-8 flex-none relative overflow-hidden"
              style={{ background: C.bgCard, minWidth: '220px', boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}
            >
              <div
                className="absolute top-0 left-0 right-0 h-1"
                style={{ background: `linear-gradient(90deg, ${C.accent}, transparent)` }}
              />
              <div className="text-5xl font-bold mb-1" style={{ ...SG, color: C.accent }}>0.002s</div>
              <div className="text-[10px] font-medium tracking-[0.2em] uppercase" style={{ color: C.textMuted }}>Identification Relay</div>
            </div>
          </div>

          {/* Architecture cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Left: Scanning grid visual + feature list */}
            <div
              className="arch-anim rounded-2xl overflow-hidden"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
            >
              {/* Grid visual */}
              <div className="relative h-52 overflow-hidden" style={{ background: C.bgOverlay }}>
                <div
                  className="absolute inset-0"
                  style={{
                    backgroundImage: `linear-gradient(rgba(39,209,127,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(39,209,127,0.1) 1px, transparent 1px)`,
                    backgroundSize: '24px 24px',
                  }}
                />
                {/* Scan line */}
                <div
                  className="absolute left-0 right-0 h-[2px]"
                  style={{
                    top: '50%',
                    background: `linear-gradient(90deg, transparent, ${C.accent}, transparent)`,
                    boxShadow: `0 0 16px rgba(39,209,127,0.4)`,
                    animation: 'scanSweepArch 3s ease-in-out infinite',
                  }}
                />
                {/* Data nodes */}
                {[
                  { top: '30%', left: '20%' },
                  { top: '60%', left: '45%' },
                  { top: '40%', left: '70%' },
                  { top: '70%', left: '85%' },
                ].map((pos, i) => (
                  <div
                    key={i}
                    className="absolute w-2.5 h-2.5 rounded-full animate-pulse"
                    style={{
                      top: pos.top, left: pos.left,
                      background: C.accent,
                      animationDelay: `${i * 0.4}s`,
                      boxShadow: `0 0 10px rgba(39,209,127,0.6)`,
                    }}
                  />
                ))}
              </div>

              {/* Feature list */}
              <div className="p-8 space-y-6">
                {[
                  { icon: Focus, title: 'Auto-Isolation', desc: 'Neural networks automatically isolate abnormal pixel clusters for focused review.' },
                  { icon: SlidersHorizontal, title: 'Dynamic Thresholding', desc: 'Sensitivity levels adapt in real-time based on the clinical history of the subject.' },
                ].map((f) => (
                  <div key={f.title} className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-none" style={{ background: 'rgba(39,209,127,0.1)' }}>
                      <f.icon className="w-5 h-5" style={{ color: C.accent }} />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold mb-1" style={{ ...SG, color: C.text }}>{f.title}</h4>
                      <p className="text-sm leading-relaxed" style={{ color: C.textSub }}>{f.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: Immutable Audit Log */}
            <div
              className="arch-anim rounded-2xl p-8"
              style={{ background: C.bgCard, boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
            >
              <div className="flex items-center gap-3 mb-8">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-none" style={{ background: 'rgba(39,209,127,0.1)' }}>
                  <FileCheck className="w-5 h-5" style={{ color: C.accent }} />
                </div>
                <div>
                  <h4 className="text-lg font-bold" style={{ ...SG, color: C.text }}>Immutable Audit Log</h4>
                  <p className="text-sm" style={{ color: C.textSub }}>Every scan is cryptographically signed and stored for total clinical accountability.</p>
                </div>
              </div>

              {/* Audit log table */}
              <div className="space-y-0">
                {/* Header */}
                <div className="grid grid-cols-4 gap-4 pb-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                  <span className="text-[10px] font-medium tracking-[0.15em] uppercase" style={{ color: C.textMuted }}>Time</span>
                  <span className="text-[10px] font-medium tracking-[0.15em] uppercase" style={{ color: C.textMuted }}>Action</span>
                  <span className="text-[10px] font-medium tracking-[0.15em] uppercase" style={{ color: C.textMuted }}>Status</span>
                  <span className="text-[10px] font-medium tracking-[0.15em] uppercase" style={{ color: C.textMuted }}>Hash</span>
                </div>
                {/* Rows */}
                {auditLog.map((row, i) => (
                  <div
                    key={i}
                    className="grid grid-cols-4 gap-4 py-4"
                    style={{ borderBottom: i < auditLog.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}
                  >
                    <span className="text-sm font-mono" style={{ color: C.textSub }}>{row.time}</span>
                    <span className="text-sm font-semibold" style={{ color: C.text }}>{row.action}</span>
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.accent, boxShadow: `0 0 6px rgba(39,209,127,0.6)` }} />
                      <span className="text-xs font-medium" style={{ color: C.accent }}>{row.status}</span>
                    </div>
                    <span className="text-xs font-mono" style={{ color: C.textMuted }}>{row.hash}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          CTA SECTION
      ═══════════════════════════════════════════════════════════════════ */}
      <section id="cta" className="relative py-28 lg:py-36" style={{ background: C.bg }}>
        <div ref={ctaRef} className="max-w-[1400px] mx-auto px-6 lg:px-12 text-center">
          <div
            className="w-16 h-16 rounded-2xl mx-auto flex items-center justify-center mb-8"
            style={{ background: 'rgba(39,209,127,0.1)' }}
          >
            <Rocket className="w-8 h-8" style={{ color: C.accent }} />
          </div>
          <h2
            className="text-4xl lg:text-6xl font-bold leading-tight mb-6"
            style={{ ...SG, color: C.text }}
          >
            Ready to upgrade<br />your <span className="italic" style={{ color: C.accent }}>clinical vision?</span>
          </h2>
          <p className="text-base lg:text-lg max-w-2xl mx-auto mb-10 leading-relaxed" style={{ color: C.textSub }}>
            Join over 450 leading medical institutions utilizing Retina AI to set
            the new standard in precision diagnostics.
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Button
              size="lg"
              className="bg-[#27D17F] hover:bg-[#22b86e] text-[#0a1e2c] font-bold px-10 rounded-xl group text-base shadow-lg shadow-[#27D17F]/20"
              onClick={() => navigate('/app')}
            >
              Get Started Free
              <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button
              size="lg"
              variant="ghost"
              className="rounded-xl font-medium"
              style={{ color: 'rgba(255,255,255,0.7)' }}
            >
              View Demo
            </Button>
          </div>

          {/* Partner logos */}
          <div className="mt-20 flex items-center justify-center gap-12 flex-wrap">
            {['MEDICORP', 'OPTIC.IO', 'NEUROLINK', 'RETINEX', 'VISIONLAB'].map((name) => (
              <span key={name} className="text-[11px] font-bold tracking-[0.2em] uppercase" style={{ color: 'rgba(255,255,255,0.2)' }}>{name}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          FOOTER
      ═══════════════════════════════════════════════════════════════════ */}
      <footer className="py-12" style={{ background: C.bgDark, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="max-w-[1400px] mx-auto px-6 lg:px-12 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <Microscope className="w-5 h-5" style={{ color: C.accent }} />
            <span className="font-bold text-sm" style={{ ...SG, color: C.text }}>RETINA AI</span>
          </div>
          <div className="flex items-center gap-8 flex-wrap justify-center">
            {['Privacy Protocol', 'Terms of Operation', 'System Status', 'Contact Support'].map((link) => (
              <a
                key={link}
                href="#"
                className="text-[10px] font-medium tracking-[0.12em] uppercase transition-colors hover:opacity-80"
                style={{ color: C.textMuted }}
              >
                {link}
              </a>
            ))}
          </div>
          <span className="text-[10px]" style={{ color: 'rgba(255,255,255,0.15)' }}>© 2025 Retina AI Labs</span>
        </div>
      </footer>

      {/* Keyframes */}
      <style>{`
        @keyframes scanSweepArch {
          0%   { top: 10%; opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { top: 90%; opacity: 0; }
        }
      `}</style>
    </div>
  );
}

function AsciiEye() {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    // Blinking logic: Wait 4s, blink (f0 -> f1 -> f2 -> f1 -> f0)
    const interval = setInterval(() => {
      setFrame(1);
      setTimeout(() => setFrame(2), 60);
      setTimeout(() => setFrame(1), 120);
      setTimeout(() => setFrame(0), 180);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const frames = [f0, f1, f2];
  const currentArt = frames[frame].join('\n');

  return (
    <pre
      className="text-[8px] sm:text-[9px] md:text-[10px] leading-[1.3] select-none pointer-events-none whitespace-pre"
      style={{
        fontFamily: 'monospace',
        letterSpacing: '0.02em',
        color: 'rgba(39,209,127,0.85)',
        textShadow: '0 0 12px rgba(39,209,127,0.5), 0 0 24px rgba(39,209,127,0.3)',
      }}
    >
      {currentArt}
    </pre>
  );
}
