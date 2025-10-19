/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"] ,
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#5B21B6",
          light: "#8B5CF6",
          dark: "#4C1D95",
        },
        accent: "#FBBF24",
      },
    },
  },
  plugins: [],
};
