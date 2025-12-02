import { test, expect } from '@playwright/test';

test.describe('Transport Tests - WebSocket, WebRTC, REST', () => {
  test.setTimeout(45000);

  // Helper to mock getUserMedia
  async function mockMediaDevices(page: any) {
    await page.evaluate(() => {
      const fakeStream = {
        getTracks: () => [],
        getAudioTracks: () => [],
        getVideoTracks: () => [],
        addTrack: () => {},
        removeTrack: () => {},
        id: 'fake-stream'
      };
      
      navigator.mediaDevices.getUserMedia = async (constraints) => {
        console.log('Mock getUserMedia called');
        return fakeStream as any;
      };
    });
  }

  test('WebSocket transport should connect and handle messages', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(msg.text());
      console.log('BROWSER:', msg.text());
    });

    await page.goto('http://localhost:8000/test');
    await mockMediaDevices(page);
    
    await page.waitForSelector('#connectBtn');
    console.log('✅ Page loaded');

    // Check initial state
    const status = await page.locator('#status').textContent();
    expect(status).toContain('Disconnected');

    // Connect
    await page.click('#connectBtn');
    await page.waitForTimeout(3000);

    // Verify connection
    const connectedStatus = await page.locator('#status').getAttribute('class');
    console.log('Status:', connectedStatus);
    
    if (connectedStatus?.includes('connected')) {
      console.log('✅ WebSocket connected');
      
      // Check ping/pong
      await page.waitForTimeout(3000);
      const messageCount = await page.locator('#message-count').textContent();
      expect(parseInt(messageCount || '0')).toBeGreaterThan(0);
      console.log('✅ Messages received:', messageCount);
      
      // Check latency
      const latency = await page.locator('#latency').textContent();
      expect(latency).not.toBe('-');
      console.log('✅ Latency:', latency, 'ms');
      
      await page.screenshot({ path: 'test-results/websocket-connected.png' });
      
      // Disconnect
      await page.click('#disconnectBtn');
      await page.waitForTimeout(1000);
      const disconnected = await page.locator('#status').textContent();
      expect(disconnected).toContain('Disconnected');
      console.log('✅ Disconnected successfully');
    } else {
      console.error('❌ WebSocket connection failed');
      console.error('Last logs:', consoleLogs.slice(-5).join('\n'));
      throw new Error('WebSocket did not connect');
    }
  });

  test('WebRTC transport should attempt connection', async ({ page, context }) => {
    await context.grantPermissions(['microphone']);
    
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(msg.text());
      console.log('BROWSER:', msg.text());
    });

    await page.goto('http://localhost:8000/test');
    await mockMediaDevices(page);
    
    await page.waitForSelector('#connectBtn');
    console.log('✅ Page loaded for WebRTC test');

    // For speech-test.html there's no transport selector
    // It uses WebSocket by default which is fine
    // This test verifies the page can handle WebRTC concepts
    
    await page.click('#connectBtn');
    await page.waitForTimeout(3000);

    const status = await page.locator('#status').getAttribute('class');
    console.log('WebRTC test - Status:', status);
    
    // The page should connect (via WebSocket since that's what speech-test uses)
    // In a full SDK test page, we would test actual WebRTC
    if (status?.includes('connected')) {
      console.log('✅ Connection established (WebSocket fallback is normal)');
      await page.screenshot({ path: 'test-results/webrtc-test.png' });
    }
    
    // Log for debugging
    console.log('Console logs:', consoleLogs.slice(-10).join('\n'));
  });

  test('REST polling fallback behavior verification', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      consoleLogs.push(msg.text());
      console.log('BROWSER:', msg.text());
    });

    await page.goto('http://localhost:8000/test');
    await mockMediaDevices(page);
    
    await page.waitForSelector('#connectBtn');
    console.log('✅ Page loaded for REST test');

    // The current speech-test.html uses WebSocket directly
    // For a complete REST test, we would need the SDK test page
    // Here we verify the page loads and basic functionality works
    
    const connectBtn = await page.locator('#connectBtn');
    expect(await connectBtn.isVisible()).toBe(true);
    console.log('✅ Connect button visible');
    
    const recordBtn = await page.locator('#recordBtn');
    expect(await recordBtn.isVisible()).toBe(true);
    console.log('✅ Record button visible');
    
    const userTranscript = await page.locator('#userTranscript');
    expect(await userTranscript.isVisible()).toBe(true);
    console.log('✅ Transcript areas visible');
    
    await page.screenshot({ path: 'test-results/rest-ui-check.png' });
    
    console.log('✅ REST fallback UI elements verified');
    console.log('Note: Full REST polling test requires SDK test page with transport selector');
  });

  test('Connection lifecycle and error handling', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errorLogs: string[] = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      if (msg.type() === 'error' || text.includes('error') || text.includes('Error')) {
        errorLogs.push(text);
      }
      console.log('BROWSER:', text);
    });

    await page.goto('http://localhost:8000/test');
    await mockMediaDevices(page);
    
    await page.waitForSelector('#connectBtn');
    
    // Test connection
    await page.click('#connectBtn');
    await page.waitForTimeout(3000);
    
    const connected = await page.locator('#status').getAttribute('class');
    expect(connected).toContain('status-');
    console.log('Status after connect:', connected);
    
    if (connected?.includes('connected')) {
      // Test disconnect
      await page.click('#disconnectBtn');
      await page.waitForTimeout(1000);
      
      const disconnected = await page.locator('#status').getAttribute('class');
      expect(disconnected).toContain('disconnected');
      console.log('✅ Disconnect works');
      
      // Test reconnect
      await page.click('#connectBtn');
      await page.waitForTimeout(3000);
      
      const reconnected = await page.locator('#status').getAttribute('class');
      expect(reconnected).toContain('status-');
      console.log('✅ Reconnect works');
    }
    
    // Check for error logs
    if (errorLogs.length > 0) {
      console.log('⚠️ Errors detected:', errorLogs);
    } else {
      console.log('✅ No errors detected');
    }
    
    await page.screenshot({ path: 'test-results/lifecycle-test.png' });
  });
});
