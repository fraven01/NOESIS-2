/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['theme/**/*.html', 'theme/**/*.tsx', 'theme/**/*.ts'],
  theme: {
    container: {
      center: true,
    },
    extend: {
      colors: {
        bg: 'rgb(var(--bg) / <alpha-value>)',
        fg: 'rgb(var(--fg) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        accent: 'rgb(var(--accent) / <alpha-value>)',
        danger: 'rgb(var(--danger) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['InterVariable', 'system-ui', 'sans-serif'],
      },
      spacing: {
        0: 'var(--space-0)',
        1: 'var(--space-1)',
        2: 'var(--space-2)',
        3: 'var(--space-3)',
        4: 'var(--space-4)',
        5: 'var(--space-5)',
        6: 'var(--space-6)',
        7: 'var(--space-7)',
        8: 'var(--space-8)',
        9: 'var(--space-9)',
        10: 'var(--space-10)',
        11: 'var(--space-11)',
        12: 'var(--space-12)',
        14: 'var(--space-14)',
        16: 'var(--space-16)',
      },
      borderRadius: {
        xl: 'var(--radius-xl)',
        '2xl': 'var(--radius-2xl)',
      },
      boxShadow: {
        sm: 'var(--shadow-sm)',
        DEFAULT: 'var(--shadow)',
        md: 'var(--shadow-md)',
        lg: 'var(--shadow-lg)',
      },
      transitionDuration: {
        150: 'var(--duration-150)',
        200: 'var(--duration-200)',
        300: 'var(--duration-300)',
      },
      transitionTimingFunction: {
        DEFAULT: 'var(--easing-standard)',
      },
      fontSize: {
        xs: ['var(--text-xs)', { lineHeight: '1.25' }],
        sm: ['var(--text-sm)', { lineHeight: '1.4' }],
        base: ['var(--text-base)', { lineHeight: '1.5' }],
        lg: ['var(--text-lg)', { lineHeight: '1.6' }],
        xl: ['var(--text-xl)', { lineHeight: '1.6' }],
        '2xl': ['var(--text-2xl)', { lineHeight: '1.2' }],
      },
    },
  },
  plugins: [],
};
