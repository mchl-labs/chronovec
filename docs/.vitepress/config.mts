import { defineConfig } from 'vitepress'

// Set base to '/chronovec/' for GitHub Pages without a custom domain.
// Change to '/' once you point a custom domain at the Pages site.
const base = '/chronovec/'

export default defineConfig({
  base,
  title: 'ChronoVec',
  description:
    'The embeddable vector index that keeps working as your data changes. Snapshot isolation, speculative branching, bounded deletion, crash recovery.',
  cleanUrls: true,
  lastUpdated: true,

  sitemap: {
    hostname: `https://mchl-labs.github.io${base}`,
  },

  head: [
    ['link', { rel: 'icon', type: 'image/png', href: `${base}chronovec-icon.png` }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'ChronoVec' }],
    ['meta', { property: 'og:description', content: 'The embeddable vector index that keeps working as your data changes.' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    // llms.txt discovery
    ['link', { rel: 'alternate', type: 'text/plain', title: 'LLMs.txt', href: `${base}llms.txt` }],
  ],

  themeConfig: {
    logo: { src: '/chronovec-icon.png', width: 28, height: 28 },

    nav: [
      { text: 'Guide', link: '/getting-started', activeMatch: '^/(getting-started|architecture|performance|contributing|migration-from-chroma)' },
      { text: 'API', link: '/api-reference' },
      { text: 'Integrations', link: '/integrations/langchain', activeMatch: '^/integrations/' },
      { text: 'Use cases', link: '/use-cases/agent-memory', activeMatch: '^/use-cases/' },
    ],

    sidebar: {
      '/': [
        {
          text: 'Guide',
          items: [
            { text: 'Getting started', link: '/getting-started' },
            { text: 'Migrating from Chroma', link: '/migration-from-chroma' },
            { text: 'Architecture', link: '/architecture' },
            { text: 'Performance', link: '/performance' },
            { text: 'Support & maturity', link: '/support-matrix' },
            { text: 'Contributing', link: '/contributing' },
          ],
        },
        {
          text: 'API reference',
          items: [
            { text: 'All classes & methods', link: '/api-reference' },
          ],
        },
        {
          text: 'Integrations',
          items: [
            { text: 'LangChain', link: '/integrations/langchain' },
            { text: 'LlamaIndex', link: '/integrations/llamaindex' },
            { text: 'LangGraph', link: '/integrations/langgraph' },
            { text: 'DuckDB', link: '/integrations/duckdb' },
            { text: 'SQLite', link: '/integrations/sqlite' },
            { text: 'Rust', link: '/integrations/rust' },
            { text: 'Go', link: '/integrations/go' },
            { text: 'Node.js', link: '/integrations/node' },
          ],
        },
        {
          text: 'Use cases',
          items: [
            { text: 'Agent memory', link: '/use-cases/agent-memory' },
            { text: 'Tree-search agents (LATS)', link: '/use-cases/lats' },
            { text: 'RAG with history', link: '/use-cases/rag-with-history' },
            { text: 'Compliance & erasure', link: '/use-cases/compliance' },
            { text: 'Streaming updates', link: '/use-cases/streaming-updates' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/mchl-labs/chronovec' },
    ],

    search: {
      provider: 'local',
    },

    editLink: {
      pattern: 'https://github.com/mchl-labs/chronovec/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },

    footer: {
      message: 'Released under the Apache-2.0 License.',
      copyright: 'Copyright © 2026 ChronoVec contributors',
    },
  },
})
