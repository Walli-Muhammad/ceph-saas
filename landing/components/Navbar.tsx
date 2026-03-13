"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const links = [
    { label: "Features", href: "#features" },
    { label: "Pricing", href: "#pricing" },
    { label: "Docs", href: "#" },
    { label: "Demo", href: "https://ceph-saas-mvp.vercel.app", external: true },
];

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener("scroll", onScroll, { passive: true });
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    return (
        <header
            className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${scrolled ? "glass border-b border-white/[0.06] shadow-lg shadow-black/20" : "bg-transparent"
                }`}
        >
            <nav className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                {/* Logo */}
                <a href="/" className="flex items-center gap-2 font-bold text-lg tracking-tight">
                    <span className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center text-white text-xs font-black">C</span>
                    <span className="text-white">Cephalo<span className="gradient-text">AI</span></span>
                </a>

                {/* Desktop links */}
                <ul className="hidden md:flex items-center gap-8 text-sm text-slate-400">
                    {links.map((l) => (
                        <li key={l.label}>
                            <a
                                href={l.href}
                                className="hover:text-white transition-colors duration-200"
                                {...(l.external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
                            >
                                {l.label}
                            </a>
                        </li>
                    ))}
                </ul>

                {/* CTA */}
                <div className="hidden md:flex items-center gap-3">
                    <a href="https://ceph-saas-mvp.vercel.app" target="_blank" rel="noopener noreferrer" className="text-sm text-slate-400 hover:text-white transition-colors px-4 py-2">
                        Log In
                    </a>
                    <a
                        href="#pricing"
                        className="text-sm font-semibold px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-teal-500 text-white hover:opacity-90 transition-opacity"
                    >
                        Get Started Free
                    </a>
                </div>

                {/* Mobile toggle */}
                <button
                    className="md:hidden text-slate-400 hover:text-white"
                    onClick={() => setMobileOpen(!mobileOpen)}
                    aria-label="Toggle menu"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        {mobileOpen
                            ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />}
                    </svg>
                </button>
            </nav>

            {/* Mobile menu */}
            <AnimatePresence>
                {mobileOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="md:hidden glass border-t border-white/[0.06] px-6 py-4 flex flex-col gap-4"
                    >
                        {links.map((l) => (
                            <a key={l.label} href={l.href} className="text-slate-300 hover:text-white text-sm">
                                {l.label}
                            </a>
                        ))}
                        <a
                            href="#pricing"
                            className="text-sm font-semibold px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-teal-500 text-white text-center"
                        >
                            Get Started Free
                        </a>
                    </motion.div>
                )}
            </AnimatePresence>
        </header>
    );
}
