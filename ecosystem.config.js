module.exports = {
  apps: [{
    name: 'osu-dreamer-server',
    cmd: 'scripts/server.py',
    args: "epoch=97-step=101446.ckpt --port 6500",
    autorestart: true,
    max_memory_restart: '6G',
    interpreter: 'python3'
  }],

  deploy: {
    production: {
      user: 'SSH_USERNAME',
      host: 'SSH_HOSTMACHINE',
      ref: 'origin/master',
      repo: 'GIT_REPOSITORY',
      path: 'DESTINATION_PATH',
      'pre-deploy-local': '',
      'post-deploy': 'npm install && pm2 reload ecosystem.config.js --env production',
      'pre-setup': ''
    }
  }
};
