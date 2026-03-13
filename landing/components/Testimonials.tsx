"use client";
import { motion } from "framer-motion";

const testimonials = [
    {
        stars: 5,
        quote:
            "CephaloAI has completely transformed how I onboard new patients. The landmark detection is eerily accurate and the reports save me 30 minutes per case.",
        name: "Dr. Sarah Mitchell",
        title: "Orthodontist, Mitchell Dental Group",
        initials: "SM",
        color: "from-cyan-500 to-blue-500",
    },
    {
        stars: 5,
        quote:
            "The AI assistant is a genuine clinical co-pilot. It flags abnormalities I might catch late and generates treatment-ready narratives in seconds.",
        name: "Dr. Raj Kumar",
        title: "Consultant Orthodontist, NHS",
        initials: "RK",
        color: "from-purple-500 to-pink-500",
    },
    {
        stars: 5,
        quote:
            "Collaborating on complex cases has never been easier. We share analyses, leave comments, and co-sign digitally — all within one platform.",
        name: "Dr. Aida Fontaine",
        title: "Principal, Fontaine Orthodontics",
        initials: "AF",
        color: "from-emerald-500 to-cyan-500",
    },
];

function Stars({ n }: { n: number }) {
    return (
        <div className="flex gap-1">
            {Array.from({ length: n }).map((_, i) => (
                <svg key={i} className="w-4 h-4 text-amber-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
            ))}
        </div>
    );
}

export default function Testimonials() {
    return (
        <section className="py-28 px-6">
            <div className="max-w-7xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 24 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.3 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    <span className="text-xs font-semibold tracking-widest uppercase text-cyan-400 mb-4 block">
                        Trusted by Leading Orthodontists
                    </span>
                    <h2 className="text-4xl font-extrabold text-white tracking-tight">
                        What clinicians are <span className="gradient-text">saying</span>
                    </h2>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {testimonials.map((t, i) => (
                        <motion.div
                            key={t.name}
                            initial={{ opacity: 0, y: 40 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true, amount: 0.2 }}
                            transition={{ duration: 0.6, delay: i * 0.12, ease: "easeOut" }}
                            whileHover={{ y: -4 }}
                            className="glass card-hover-glow rounded-2xl p-7 flex flex-col gap-5 border border-white/[0.07]"
                        >
                            <Stars n={t.stars} />
                            <p className="text-slate-300 text-sm leading-relaxed flex-1">&ldquo;{t.quote}&rdquo;</p>
                            <div className="flex items-center gap-3 pt-2 border-t border-white/[0.06]">
                                <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${t.color} flex items-center justify-center text-white text-xs font-bold flex-shrink-0`}>
                                    {t.initials}
                                </div>
                                <div>
                                    <p className="text-white text-sm font-semibold">{t.name}</p>
                                    <p className="text-slate-500 text-xs">{t.title}</p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
