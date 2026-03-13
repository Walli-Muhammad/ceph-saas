"use client";
import { useState, useRef, MouseEvent } from "react";
import { motion } from "framer-motion";

interface Plan {
    name: string;
    price: string;
    period: string;
    badge?: string;
    features: string[];
    cta: string;
    highlighted: boolean;
}

const plans: Plan[] = [
    {
        name: "Starter",
        price: "$99",
        period: "/month",
        features: [
            "50 analyses per month",
            "Automatic landmark detection",
            "Standard PDF reports",
            "Email support",
            "JPEG & PNG upload",
        ],
        cta: "Get Started",
        highlighted: false,
    },
    {
        name: "Professional",
        price: "$249",
        period: "/month",
        badge: "Most Popular",
        features: [
            "Unlimited analyses",
            "AI clinical summaries",
            "Editable landmark handles",
            "Collaborative review",
            "Priority support",
            "DICOM support",
            "Custom calibration",
        ],
        cta: "Start Free Trial",
        highlighted: true,
    },
    {
        name: "Enterprise",
        price: "Custom",
        period: "",
        features: [
            "Everything in Professional",
            "Dedicated infrastructure",
            "SSO & audit logs",
            "Custom integrations",
            "SLA guarantee",
            "Onboarding & training",
        ],
        cta: "Contact Sales",
        highlighted: false,
    },
];

function PricingCard({ plan, i }: { plan: Plan; i: number }) {
    const cardRef = useRef<HTMLDivElement>(null);
    const [spotlight, setSpotlight] = useState({ x: 0, y: 0, show: false });

    const onMouseMove = (e: MouseEvent<HTMLDivElement>) => {
        const rect = cardRef.current?.getBoundingClientRect();
        if (!rect) return;
        setSpotlight({ x: e.clientX - rect.left, y: e.clientY - rect.top, show: true });
    };

    return (
        <motion.div
            ref={cardRef}
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.6, delay: i * 0.12, ease: "easeOut" }}
            onMouseMove={onMouseMove}
            onMouseLeave={() => setSpotlight((s) => ({ ...s, show: false }))}
            className={`relative rounded-2xl p-8 flex flex-col gap-6 border overflow-hidden transition-all duration-300 cursor-default
        ${plan.highlighted
                    ? "border-cyan-500/50 bg-gradient-to-b from-cyan-950/40 to-purple-950/20 shadow-2xl shadow-cyan-500/10"
                    : "glass border-white/[0.07] card-hover-glow"}
      `}
        >
            {/* Spotlight */}
            {spotlight.show && (
                <div
                    className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-200"
                    style={{
                        background: `radial-gradient(300px circle at ${spotlight.x}px ${spotlight.y}px, rgba(34,211,238,0.08), transparent 70%)`,
                        opacity: spotlight.show ? 1 : 0,
                    }}
                />
            )}

            {/* Badge */}
            {plan.badge && (
                <span className="absolute top-5 right-5 text-[10px] font-bold tracking-wider uppercase px-2.5 py-1 rounded-full bg-gradient-to-r from-cyan-500 to-purple-600 text-white">
                    {plan.badge}
                </span>
            )}

            <div>
                <p className="text-sm font-semibold text-slate-400 mb-3">{plan.name}</p>
                <div className="flex items-end gap-1">
                    <span className="text-5xl font-black text-white tracking-tight">{plan.price}</span>
                    {plan.period && <span className="text-slate-500 text-sm pb-2">{plan.period}</span>}
                </div>
            </div>

            <ul className="flex flex-col gap-3 flex-1">
                {plan.features.map((f) => (
                    <li key={f} className="flex items-start gap-3 text-sm text-slate-300">
                        <svg className="w-4 h-4 mt-0.5 text-cyan-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                        </svg>
                        {f}
                    </li>
                ))}
            </ul>

            <a
                href="#"
                className={`text-center text-sm font-semibold py-3 px-6 rounded-xl transition-all active:scale-95
          ${plan.highlighted
                        ? "bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:opacity-90 shadow-lg shadow-purple-500/20"
                        : "glass border border-white/10 text-white hover:border-white/20"}
        `}
            >
                {plan.cta}
            </a>
        </motion.div>
    );
}

export default function Pricing() {
    return (
        <section id="pricing" className="py-28 px-6 relative overflow-hidden">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-[300px] glow-purple opacity-10 blur-3xl pointer-events-none" />

            <div className="max-w-5xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 24 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.3 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    <span className="text-xs font-semibold tracking-widest uppercase text-purple-400 mb-4 block">
                        Simple, Transparent Pricing
                    </span>
                    <h2 className="text-4xl font-extrabold text-white tracking-tight mb-3">
                        Choose your <span className="gradient-text">plan</span>
                    </h2>
                    <p className="text-slate-400 text-sm">No hidden fees. Cancel anytime. 14-day free trial on all plans.</p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-stretch">
                    {plans.map((plan, i) => (
                        <PricingCard key={plan.name} plan={plan} i={i} />
                    ))}
                </div>
            </div>
        </section>
    );
}
