/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'surface': {
          DEFAULT: '#13161d',
          'deep': '#0d0f14',
        },
        'primary': '#6e7bf7',
        'accent-violet': '#9b87f5',
        'accent-rose': '#e8729a',
        'accent-green': '#2ec98a',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['Fira Code', 'monospace'],
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
