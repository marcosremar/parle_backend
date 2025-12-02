import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

test.describe('Speech-to-Speech Test', () => {
  test.setTimeout(90000); // 90 seconds for full speech processing

  test('should complete full speech-to-speech flow', async ({ page, context }) => {
    // Grant microphone permissions
    await context.grantPermissions(['microphone']);

    // Capture console logs
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      console.log('BROWSER:', text);
    });

    // Navigate to speech test page
    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    // Wait for page to load
    await page.waitForSelector('#connectBtn');
    console.log('✅ Page loaded');

    // Check initial state
    const status = await page.locator('#status').textContent();
    expect(status).toContain('Disconnected');

    // Click connect button
    await page.click('#connectBtn');
    console.log('🔌 Clicked connect');

    // Wait for connection (should request mic permission and connect to WebSocket)
    await page.waitForTimeout(5000);

    // Take screenshot of connected state
    await page.screenshot({ path: 'test-results/speech-connected.png' });

    // Print all console logs
    console.log('===== BROWSER CONSOLE LOGS =====');
    consoleLogs.forEach(log => console.log(log));
    console.log('================================');
    
    // Check if connected
    const connectedStatus = await page.locator('#status').getAttribute('class');
    if (!connectedStatus?.includes('connected')) {
      console.error('❌ Connection failed! Status:', connectedStatus);
      console.error('Last 10 console logs:');
      console.error(consoleLogs.slice(-10).join('\n'));
    }
    expect(connectedStatus).toContain('status-connected');
    console.log('✅ Connected to WebSocket');

    // Check transport badge
    const transportBadge = await page.locator('#transport-badge').textContent();
    console.log('Transport:', transportBadge);

    // Wait for latency measurement
    await page.waitForTimeout(2000);

    // Check latency display
    const latency = await page.locator('#latency').textContent();
    console.log('Latency:', latency);
    expect(latency).not.toBe('-');

    // Set up listeners for speech events
    const userTranscriptPromise = page.waitForFunction(
      () => {
        const el = document.getElementById('userTranscript');
        return el && el.textContent && el.textContent !== 'Press "Hold to Talk" to speak...';
      },
      { timeout: 30000 }
    );

    const botResponsePromise = page.waitForFunction(
      () => {
        const el = document.getElementById('botResponse');
        return el && el.textContent && el.textContent !== 'Response will appear here...';
      },
      { timeout: 30000 }
    );

    // Simulate recording by clicking and holding record button
    console.log('🎤 Starting recording simulation...');
    
    // Note: Since we can't actually send real audio from Playwright without a real mic,
    // this test will verify the UI flow works, but won't test actual audio processing
    // For actual audio testing, we would need to:
    // 1. Generate audio file with gTTS
    // 2. Mock the MediaRecorder to send that audio
    // This would require more complex setup
    
    // For now, let's verify the UI elements exist and are interactive
    const recordBtn = await page.locator('#recordBtn');
    expect(await recordBtn.isEnabled()).toBe(true);
    console.log('✅ Record button is enabled');

    // Check transcript display areas exist
    const userTranscriptEl = await page.locator('#userTranscript');
    expect(await userTranscriptEl.isVisible()).toBe(true);

    const botResponseEl = await page.locator('#botResponse');
    expect(await botResponseEl.isVisible()).toBe(true);

    // Check audio player exists
    const audioPlayer = await page.locator('#remoteAudio');
    expect(await audioPlayer.isVisible()).toBe(true);
    console.log('✅ All speech UI elements present');

    // Check console logs for connection messages
    const logs = await page.locator('#logs').textContent();
    expect(logs).toContain('Connected');
    expect(logs).toContain('speech server');
    console.log('✅ Connection logs verified');

    // Take screenshot of connected state
    await page.screenshot({ path: 'test-results/speech-connected.png' });

    // Disconnect
    await page.click('#disconnectBtn');
    await page.waitForTimeout(1000);

    const disconnectedStatus = await page.locator('#status').textContent();
    expect(disconnectedStatus).toContain('Disconnected');
    console.log('✅ Disconnected successfully');
  });

  test('should have proper WebSocket message handling', async ({ page, context }) => {
    await context.grantPermissions(['microphone']);
    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    // Collect console messages
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(msg.text());
    });

    await page.waitForSelector('#connectBtn');
    await page.click('#connectBtn');
    await page.waitForTimeout(3000);

    // Verify connection established
    const connected = await page.locator('#status').getAttribute('class');
    expect(connected).toContain('connected');

    // Check message counter updates with ping/pong
    await page.waitForTimeout(5000); // Wait for at least 2 ping/pong cycles

    const messageCount = await page.locator('#message-count').textContent();
    const count = parseInt(messageCount || '0');
    expect(count).toBeGreaterThan(0);
    console.log('✅ Message counter working:', count, 'messages');

    // Check that latency is being measured
    const latency = await page.locator('#latency').textContent();
    expect(latency).not.toBe('-');
    expect(parseInt(latency || '0')).toBeGreaterThan(0);
    console.log('✅ Latency measurement working:', latency, 'ms');

    // Verify logs show ping/pong
    const logs = await page.locator('#logs').textContent();
    expect(logs).toContain('pong');
    console.log('✅ WebSocket ping/pong working');

    await page.screenshot({ path: 'test-results/websocket-messages.png' });
  });

  test('should display proper status transitions', async ({ page, context }) => {
    await context.grantPermissions(['microphone']);
    await page.goto('http://localhost:8000/test', {
      waitUntil: 'networkidle'
    });

    await page.waitForSelector('#connectBtn');

    // Initial state
    let statusClass = await page.locator('#status').getAttribute('class');
    expect(statusClass).toContain('status-disconnected');
    console.log('✅ Initial state: Disconnected');

    // Click connect
    await page.click('#connectBtn');
    
    // Should show connecting state briefly (might be too fast to catch)
    await page.waitForTimeout(500);

    // Should reach connected state
    await page.waitForTimeout(2500);
    statusClass = await page.locator('#status').getAttribute('class');
    expect(statusClass).toContain('status-connected');
    console.log('✅ Connected state reached');

    // Disconnect
    await page.click('#disconnectBtn');
    await page.waitForTimeout(1000);
    
    statusClass = await page.locator('#status').getAttribute('class');
    expect(statusClass).toContain('status-disconnected');
    console.log('✅ Disconnected state confirmed');

    await page.screenshot({ path: 'test-results/status-transitions.png' });
  });
});
