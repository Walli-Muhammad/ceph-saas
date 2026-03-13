"use client";
import { motion } from "framer-motion";
import { ReactNode } from "react";

interface Feature {
    icon: ReactNode;
    title: string;
    desc: string;
    area: string;
    accent: string;
}

const features: Feature[] = [
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
        ),
        title: "Automatic Landmark Detection",
        desc: "Our deep learning model identifies 30+ anatomical landmarks on cephalometric X-rays with sub-millimeter precision — instantly, every time.",
        area: "landmark",
        accent: "from-cyan-500 to-cyan-400",
    },
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 11h.01M12 11h.01M15 11h.01M4 19h16a2 2 0 002-2V7a2 2 0 00-2-2H4a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
        ),
        title: "Instant Measurements",
        desc: "SNA, SNB, ANB, Go-Gn-Sn, and 20+ other critical angles and distances computed in milliseconds after landmark placement.",
        area: "measurements",
        accent: "from-purple-500 to-purple-400",
    },
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
        ),
        title: "Comprehensive Reports",
        desc: "Export full cephalometric PDF reports with annotated images, measurement tables, and AI-generated clinical narratives — ready to share.",
        area: "reports",
        accent: "from-emerald-500 to-cyan-500",
    },
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
            </svg>
        ),
        title: "Treatment Planning",
        desc: "Overlay target norms, visualise discrepancies and simulate treatment outcomes with color-coded severity gauges for each measurement.",
        area: "treatment",
        accent: "from-pink-500 to-purple-500",
    },
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
        ),
        title: "Multiformat Support",
        desc: "Upload JPEG, PNG, DICOM previews and more. Our pipeline handles any digital cephalogram with automatic quality checks.",
        area: "format",
        accent: "from-amber-400 to-orange-500",
    },
    {
        icon: (
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
        ),
        title: "Collaborative Review",
        desc: "Invite colleagues to review analyses, leave comments, and co-sign treatment plans — seamless multi-practitioner workflow.",
        area: "collab",
        accent: "from-indigo-500 to-purple-500",
    },
];


export default function Features() {
    return (
        <section id="features" className="py-28 px-6 relative overflow-hidden">
            {/* Background glow */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[400px] glow-purple opacity-10 blur-3xl pointer-events-none" />

            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 24 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.3 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    <span className="text-xs font-semibold tracking-widest uppercase text-cyan-400 mb-4 block">
                        Intelligent Analysis Features
                    </span>
                    <h2 className="text-4xl lg:text-5xl font-extrabold tracking-tight text-white mb-4">
                        Everything a clinician needs,{" "}
                        <span className="gradient-text">nothing they don&apos;t.</span>
                    </h2>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                        Purpose-built for orthodontists. Powered by deep learning trained on hundreds of thousands of verified cephalometric analyses.
                    </p>
                </motion.div>

                {/* Bento Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-3 gap-4 auto-rows-fr">
                    {features.map((f, i) => {
                        const isLarge = i === 0 || i === 2; // span 2 cols on desktop for visual interest
                        return (
                            <motion.div
                                key={f.title}
                                initial={{ opacity: 0, y: 40 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true, amount: 0.2 }}
                                transition={{ duration: 0.6, delay: i * 0.1, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] }}
                                whileHover={{ y: -4 }}
                                className={`glass card-hover-glow rounded-2xl p-7 flex flex-col gap-5 border border-white/[0.07] cursor-default
                  ${isLarge ? "md:col-span-2 lg:col-span-2" : ""}
                  ${i === 1 ? "row-span-2" : ""}
                `}
                            >
                                {/* Icon */}
                                <motion.div
                                    whileHover={{ scale: 1.12, rotate: 4 }}
                                    transition={{ type: "spring", stiffness: 300 }}
                                    className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.accent} bg-opacity-10 flex items-center justify-center text-white`}
                                    style={{ background: `linear-gradient(135deg, ${f.accent.split(" ")[0].replace("from-", "")}, ${f.accent.split(" ")[1].replace("to-", "")})`.replace("from-", "").replace("to-", "") }}
                                >
                                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-white bg-gradient-to-br ${f.accent}`}>
                                        {f.icon}
                                    </div>
                                </motion.div>

                                <div>
                                    <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
                                    <p className="text-slate-400 text-sm leading-relaxed">{f.desc}</p>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </section>
    );
}
