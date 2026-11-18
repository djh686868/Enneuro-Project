(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim() || '#2563eb';
  var accent2 = style.getPropertyValue('--accent2').trim() || '#7c3aed';
  var ink = style.getPropertyValue('--ink').trim() || '#1a1a2e';
  var muted = style.getPropertyValue('--muted').trim() || '#6b7280';
  var rule = style.getPropertyValue('--rule').trim() || '#e2e5ee';
  var bg2 = style.getPropertyValue('--bg2').trim() || '#ffffff';

  // ================================================================
  // Chart 1: LiveBench Top 15 Horizontal Bar
  // ================================================================
  (function() {
    var el = document.getElementById('chart-livebench');
    if (!el) return;
    var chart = echarts.init(el, null, { renderer: 'svg' });

    var models = [
      'GPT-5.6 Sol Max', 'Claude Fable 5 Max', 'GPT-5.5 xHigh',
      'GPT-5.6 Terra Max', 'Claude Opus 4.8 xHigh', 'Kimi K3',
      'GPT-5.4 xHigh', 'Gemini 3.1 Pro Preview', 'Claude Sonnet 5 xHigh',
      'Gemini 3.5 Flash High', 'GLM-5.2', 'Qwen 3.7 Max',
      'DeepSeek V4 Pro', 'Kimi K2.6 Thinking', 'DeepSeek V4 Flash'
    ];
    var scores = [82.4, 80.8, 79.9, 79.8, 78.9, 78.5, 78.0, 77.1, 74.8, 74.6, 73.2, 73.1, 71.6, 70.5, 65.5];
    // Color: accent for international, accent2 for China
    var colors = [
      accent, accent, accent, accent, accent, accent2,
      accent, accent, accent, accent, accent2, accent2,
      accent2, accent2, accent2
    ];

    chart.setOption({
      animation: false,
      grid: { left: 180, right: 40, top: 20, bottom: 20 },
      xAxis: { type: 'value', name: 'LiveBench 综合分', nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'category', data: models.slice().reverse(), axisLabel: { color: ink, fontSize: 12 }, axisLine: { lineStyle: { color: rule } }, inverse: true },
      series: [{
        type: 'bar', data: scores.slice().reverse().map(function(v, i) { return { value: v, itemStyle: { color: colors.slice().reverse()[i], borderRadius: [0, 4, 4, 0] } }; }).reverse(),
        label: { show: true, position: 'right', color: ink, fontSize: 11, fontWeight: 700, fontFamily: 'JetBrainsMono,Consolas,monospace' },
        barMaxWidth: 24
      }],
      tooltip: { trigger: 'axis', appendToBody: true, formatter: function(p) { return p[0].name + '<br/>LiveBench: <b>' + p[0].value + '</b>'; } }
    });

    window.addEventListener('resize', function() { chart.resize(); });
  })();

  // ================================================================
  // Chart 2: API Pricing Horizontal Bar
  // ================================================================
  (function() {
    var el = document.getElementById('chart-pricing');
    if (!el) return;
    var chart = echarts.init(el, null, { renderer: 'svg' });

    var models = [
      'GPT-5.5', 'Claude Opus 4.5', 'GPT-5.4',
      'Claude Sonnet 4.6', 'Kimi K3', 'Gemini 3 Pro',
      'Gemini 3.5 Flash', 'GLM-5.2', 'GPT-5.4-mini',
      'DeepSeek V4 Pro', 'DeepSeek V4 Flash'
    ];
    var inputPrices = [5.00, 5.00, 2.50, 3.00, 3.00, 2.00, 1.50, 1.40, 0.75, 0.435, 0.14];
    var outputPrices = [30.00, 25.00, 15.00, 15.00, 15.00, 12.00, 9.00, 4.40, 4.50, 0.87, 0.28];

    chart.setOption({
      animation: false,
      grid: { left: 155, right: 120, top: 20, bottom: 20 },
      legend: { data: ['输入 ($/M)', '输出 ($/M)'], top: 0, textStyle: { color: muted, fontSize: 12 } },
      xAxis: { type: 'value', name: '$ / 百万 tokens', nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'category', data: models.slice().reverse(), axisLabel: { color: ink, fontSize: 12 }, axisLine: { lineStyle: { color: rule } }, inverse: true },
      series: [
        {
          name: '输入 ($/M)', type: 'bar',
          data: inputPrices.slice().reverse().map(function(v) { return { value: v, itemStyle: { color: accent } }; }).reverse(),
          barMaxWidth: 18, barGap: '20%',
          label: { show: true, position: 'right', color: accent, fontSize: 10, fontFamily: 'JetBrainsMono,Consolas,monospace', formatter: function(p) { return '$' + p.value; } }
        },
        {
          name: '输出 ($/M)', type: 'bar',
          data: outputPrices.slice().reverse().map(function(v) { return { value: v, itemStyle: { color: accent2, borderRadius: [0, 4, 4, 0] } }; }).reverse(),
          barMaxWidth: 18,
          label: { show: true, position: 'right', color: accent2, fontSize: 10, fontFamily: 'JetBrainsMono,Consolas,monospace', formatter: function(p) { return '$' + p.value; } }
        }
      ],
      tooltip: { trigger: 'axis', appendToBody: true, formatter: function(p) { return p[0].name + '<br/>' + p[0].seriesName + ': <b>$' + p[0].value + '</b><br/>' + p[1].seriesName + ': <b>$' + p[1].value + '</b>'; } }
    });

    window.addEventListener('resize', function() { chart.resize(); });
  })();

  // ================================================================
  // Chart 3: Performance vs Price Scatter
  // ================================================================
  (function() {
    var el = document.getElementById('chart-value');
    if (!el) return;
    var chart = echarts.init(el, null, { renderer: 'svg' });

    // Data: [model, price label, AA Intelligence Index, input price $/M]
    var data = [
      { name: 'Claude Opus 5', label: '$2.34', value: [63, 2.34], cat: 'intl', highlight: true },
      { name: 'Claude Fable 5', label: '$3.14', value: [62, 3.14], cat: 'intl', highlight: false },
      { name: 'Grok 4.6', label: '$0.94', value: [61, 0.94], cat: 'intl', highlight: false },
      { name: 'GPT-5.6 Sol', label: '$1.01', value: [61, 1.01], cat: 'intl', highlight: true },
      { name: 'Kimi K3', label: '$0.84', value: [60, 0.84], cat: 'cn', highlight: true },
      { name: 'GLM-5.3', label: '$0.68', value: [60, 0.68], cat: 'cn', highlight: true },
      { name: 'Qwen3.8 Max', label: '$0.91', value: [58, 0.91], cat: 'cn', highlight: false },
      { name: 'GLM-5.3 Flash', label: '$0.15', value: [57, 0.15], cat: 'cn', highlight: true },
      { name: 'GPT-5.6 Terra', label: '$0.53', value: [57, 0.53], cat: 'intl', highlight: false },
      { name: 'Qwen3.8 Flash-Next', label: '$0.16', value: [56, 0.16], cat: 'cn', highlight: false },
      { name: 'Gemini 3.7 Flash', label: '$0.40', value: [56, 0.40], cat: 'intl', highlight: false },
      { name: 'GLM-5.2', label: '$1.40', value: [53, 1.40], cat: 'cn', highlight: false },
      { name: 'DeepSeek V4 Pro', label: '$0.27', value: [53, 0.27], cat: 'cn', highlight: true },
      { name: 'DeepSeek V4 Flash', label: '$0.11', value: [52, 0.11], cat: 'cn', highlight: true },
      { name: 'GPT-5.6 Luna', label: '$0.05', value: [52, 0.05], cat: 'intl', highlight: false },
      { name: 'DeepSeek V4 Flash Vision', label: '$0.12', value: [51, 0.12], cat: 'cn', highlight: false },
      { name: 'MiniMax-M3', label: '$0.14', value: [45, 0.14], cat: 'cn', highlight: false }
    ];

    var intlData = data.filter(function(d) { return d.cat === 'intl'; });
    var cnData = data.filter(function(d) { return d.cat === 'cn'; });

    chart.setOption({
      animation: false,
      grid: { left: 70, right: 30, top: 20, bottom: 40 },
      xAxis: { type: 'value', name: 'AA 智能指数 →', min: 42, max: 66, nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'value', name: '输入单价 ↓ ($/M)', min: 0, max: 4, nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitLine: { lineStyle: { color: rule } } },
      series: [
        {
          name: '国际厂商', type: 'scatter',
          data: intlData.map(function(d) { return d.value; }),
          symbolSize: intlData.map(function(d) { return d.highlight ? 14 : 9; }),
          itemStyle: { color: accent, shadowBlur: 3, shadowColor: accent + '44' },
          label: {
            show: true, position: 'right', fontSize: 11, color: ink,
            formatter: function(p) { return p.seriesIndex === 0 ? intlData[p.dataIndex].label : cnData[p.dataIndex].label; }
          }
        },
        {
          name: '国产厂商', type: 'scatter',
          data: cnData.map(function(d) { return d.value; }),
          symbolSize: cnData.map(function(d) { return d.highlight ? 16 : 10; }),
          itemStyle: { color: accent2, shadowBlur: 4, shadowColor: accent2 + '55' },
          label: {
            show: true, position: 'top', fontSize: 11, color: accent2, fontWeight: 700,
            formatter: function(p) { return cnData[p.dataIndex].label; }
          }
        }
      ],
      tooltip: {
        trigger: 'item', appendToBody: true,
        formatter: function(p) {
          var arr = p.seriesIndex === 0 ? intlData : cnData;
          var d = arr[p.dataIndex];
          return '<b>' + d.name + '</b><br/>LiveBench: <b>' + d.value[0] + '</b><br/>输入: <b>' + d.label + '/M</b>';
        }
      }
    });

    window.addEventListener('resize', function() { chart.resize(); });
  })();

  // ================================================================
  // Chart 4: Reviewer candidate session cost comparison
  // ================================================================
  (function() {
    var el = document.getElementById('chart-reviewer');
    if (!el) return;
    var chart = echarts.init(el, null, { renderer: 'svg' });

    var models = ['GLM-5.3 Flash (5折)', 'GLM-5.3 Flash (标价)', 'Qwen3.8-Flash', 'MiniMax M3', 'GLM-5.2', 'Kimi K3'];
    var inputCost = [0.024, 0.048, 0.06, 0.25, 0.48, 1.20];   // 6万输入，不含缓存
    var outputCost = [0.028, 0.056, 0.06, 0.34, 0.56, 2.00];  // 2万输出
    var total = [0.052, 0.104, 0.12, 0.59, 1.04, 3.20];

    chart.setOption({
      animation: false,
      grid: { left: 155, right: 80, top: 30, bottom: 20 },
      legend: { data: ['输入成本', '输出成本', '总成本'], top: 0, textStyle: { color: muted, fontSize: 12 } },
      xAxis: { type: 'value', name: '人民币 (¥)', nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'category', data: models.slice().reverse(), axisLabel: { color: ink, fontSize: 12 }, axisLine: { lineStyle: { color: rule } }, inverse: true },
      series: [
        {
          name: '输入成本', type: 'bar',
          data: inputCost.slice().reverse().map(function(v) { return { value: v, itemStyle: { color: accent } }; }).reverse(),
          barMaxWidth: 16,
          label: { show: true, position: 'right', color: accent, fontSize: 10, formatter: function(p) { return '¥' + p.value.toFixed(2); } }
        },
        {
          name: '输出成本', type: 'bar',
          data: outputCost.slice().reverse().map(function(v) { return { value: v, itemStyle: { color: accent2 } }; }).reverse(),
          barMaxWidth: 16,
          label: { show: true, position: 'right', color: accent2, fontSize: 10, formatter: function(p) { return '¥' + p.value.toFixed(2); } }
        },
        {
          name: '总成本', type: 'bar',
          data: total.slice().reverse().map(function(v) { return { value: v, itemStyle: { color: '#f59e0b', borderRadius: [0, 4, 4, 0] } }; }).reverse(),
          barMaxWidth: 16, barGap: '30%',
          label: { show: true, position: 'right', color: '#b45309', fontSize: 11, fontWeight: 700, formatter: function(p) { return '¥' + p.value.toFixed(2); } }
        }
      ],
      tooltip: { trigger: 'axis', appendToBody: true, formatter: function(p) { return p[0].name + '<br/>' + p[0].seriesName + ': <b>¥' + p[0].value.toFixed(2) + '</b><br/>' + p[1].seriesName + ': <b>¥' + p[1].value.toFixed(2) + '</b><br/>' + p[2].seriesName + ': <b>¥' + p[2].value.toFixed(2) + '</b>'; } }
    });

    window.addEventListener('resize', function() { chart.resize(); });
  })();

})();
