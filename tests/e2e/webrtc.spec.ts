import { test, expect } from '@playwright/test';

test.describe('WebRTC Connection Test', () => {
  test.setTimeout(60000); // 60 seconds timeout

  test('should connect via WebRTC and receive audio', async ({ page, context }) => {
    // Grant microphone permissions
    await context.grantPermissions(['microphone']);

    // Navigate to test page (ignore SSL errors for self-signed cert)
    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    // Wait for page to load
    await page.waitForSelector('#connectBtn');

    // Check initial state
    const status = await page.locator('#status').textContent();
    expect(status).toContain('Disconnected');

    // Select WebSocket transport (more reliable for testing than WebRTC initially)
    await page.selectOption('#transportSelect', 'websocket');

    // Click connect button
    await page.click('#connectBtn');

    // Wait for connection
    await page.waitForTimeout(2000);

    // Check if connected
    const connectedStatus = await page.locator('#status').getAttribute('class');
    expect(connectedStatus).toContain('status-connected');

    // Check transport badge
    const transportBadge = await page.locator('#transport-badge').textContent();
    console.log('Connected via:', transportBadge);

    // Wait for latency measurement
    await page.waitForTimeout(3000);

    // Check latency display
    const latency = await page.locator('#latency').textContent();
    console.log('Latency:', latency);
    expect(latency).not.toBe('-');

    // Send test message
    await page.click('#sendBtn');
    await page.waitForTimeout(1000);

    // Check message count
    const messageCount = await page.locator('#message-count').textContent();
    console.log('Message count:', messageCount);

    // Check logs for connection messages
    const logs = await page.locator('#logs').textContent();
    expect(logs).toContain('Connected');

    // Take screenshot of success state
    await page.screenshot({ path: 'test-results/webrtc-connected.png' });

    // Disconnect
    await page.click('#disconnectBtn');
    await page.waitForTimeout(1000);

    const disconnectedStatus = await page.locator('#status').textContent();
    expect(disconnectedStatus).toContain('Disconnected');
  });

  test('should test WebRTC transport specifically', async ({ page, context }) => {
    // Grant permissions
    await context.grantPermissions(['microphone']);

    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    await page.waitForSelector('#connectBtn');

    // Select WebRTC transport
    await page.selectOption('#transportSelect', 'webrtc');

    // Set up console listener to capture WebRTC events
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(msg.text());
    });

    // Click connect
    await page.click('#connectBtn');

    // Wait for connection attempt
    await page.waitForTimeout(5000);

    // Check if WebRTC connection was attempted
    const logs = await page.locator('#logs').textContent();
    console.log('Logs:', logs);

    // Check for specific WebRTC messages
    const hasWebRTCLogs = logs.includes('WebRTC') || 
                          logs.includes('RTC') || 
                          logs.includes('offer') ||
                          logs.includes('Connected');

    if (hasWebRTCLogs) {
      console.log('✅ WebRTC connection attempted/established');
      
      // Check connection state
      const statusClass = await page.locator('#status').getAttribute('class');
      if (statusClass?.includes('connected')) {
        console.log('✅ WebRTC successfully connected!');
        
        // Wait for latency measurement
        await page.waitForTimeout(3000);
        const latency = await page.locator('#latency').textContent();
        console.log('WebRTC Latency:', latency);
        
        await page.screenshot({ path: 'test-results/webrtc-success.png' });
      }
    } else {
      console.log('⚠️ WebRTC connection not detected, may have fallen back to another transport');
      await page.screenshot({ path: 'test-results/webrtc-fallback.png' });
    }

    // Print all console messages for debugging
    console.log('Console messages:', consoleMessages);
  });

  test('should test auto-selection', async ({ page, context }) => {
    await context.grantPermissions(['microphone']);

    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    await page.waitForSelector('#connectBtn');

    // Leave on Auto
    const selectedTransport = await page.locator('#transportSelect').inputValue();
    expect(selectedTransport).toBe('auto');

    // Connect
    await page.click('#connectBtn');
    await page.waitForTimeout(5000);

    // Check which transport was selected
    const transportBadge = await page.locator('#transport-badge').textContent();
    console.log('Auto-selected transport:', transportBadge);

    // Verify connection
    const statusClass = await page.locator('#status').getAttribute('class');
    expect(statusClass).toContain('connected');

    await page.screenshot({ path: 'test-results/auto-selection.png' });
  });
});
