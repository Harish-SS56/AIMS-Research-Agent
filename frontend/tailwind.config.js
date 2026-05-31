/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          50:  '#eef2f9',
          100: '#d6e0f0',
          200: '#adc1e1',
          300: '#84a2d2',
          400: '#3d72b5',
          500: '#1B5FAA',
          600: '#175299',
          700: '#124588',
          800: '#0d3877',
          900: '#0D2D5E',
          950: '#081c3d',
        },
        amber: {
          50:  '#fff8f0',
          100: '#ffedd5',
          200: '#fed7aa',
          400: '#fb923c',
          500: '#F07920',
          600: '#D96A1A',
          700: '#c25d14',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.04)',
        'card-hover': '0 4px 12px 0 rgba(0,0,0,0.08), 0 2px 4px -2px rgba(0,0,0,0.04)',
      },
    },
  },
  plugins: [],
}
