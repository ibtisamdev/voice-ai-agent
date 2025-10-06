#!/bin/bash
# Security Hardening Script for Voice AI Agent - Hetzner VPS
# Implements comprehensive security measures for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Voice AI Agent - Security Hardening Script${NC}"
echo "============================================="
echo "Timestamp: $(date)"
echo ""

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error_exit "This script must be run as root or with sudo"
    fi
}

# Function to backup configuration file
backup_config() {
    local file="$1"
    if [ -f "$file" ]; then
        cp "$file" "${file}.backup.$(date +%Y%m%d_%H%M%S)"
        log "Backed up $file"
    fi
}

check_root

log "Starting security hardening for Hetzner VPS..."

# 1. System Updates
log "1. Updating system packages..."
apt-get update
apt-get upgrade -y
apt-get autoremove -y
apt-get autoclean

# 2. Install Security Tools
log "2. Installing security tools..."
apt-get install -y \
    ufw \
    fail2ban \
    aide \
    rkhunter \
    chkrootkit \
    lynis \
    logwatch \
    unattended-upgrades \
    apt-listchanges

# 3. Configure UFW Firewall
log "3. Configuring UFW firewall..."

# Reset UFW to defaults
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (configure your SSH port)
SSH_PORT=${SSH_PORT:-22}
ufw allow "$SSH_PORT"/tcp comment 'SSH'

# Allow HTTP and HTTPS
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Allow monitoring ports (restrict to specific IPs if needed)
ufw allow 3000/tcp comment 'Grafana'
ufw allow 9090/tcp comment 'Prometheus'

# Allow Docker network communication
ufw allow from 172.17.0.0/16
ufw allow from 172.20.0.0/16

# Enable UFW
ufw --force enable

log "UFW firewall configured and enabled"

# 4. Configure Fail2Ban
log "4. Configuring Fail2Ban..."

backup_config "/etc/fail2ban/jail.local"

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
# Ban time: 1 hour
bantime = 3600

# Find time: 10 minutes
findtime = 600

# Max retry: 3 attempts
maxretry = 3

# Ignore private IP ranges
ignoreip = 127.0.0.1/8 ::1 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16

# Email notifications (configure SMTP if needed)
destemail = admin@yourdomain.com
sender = fail2ban@yourdomain.com
mta = sendmail

# Actions
action = %(action_mwl)s

[sshd]
enabled = true
port = $SSH_PORT
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-noscript]
enabled = true
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 6

[nginx-badbots]
enabled = true
filter = nginx-badbots
logpath = /var/log/nginx/access.log
maxretry = 2

[nginx-noproxy]
enabled = true
filter = nginx-noproxy
logpath = /var/log/nginx/access.log
maxretry = 2

# Custom filter for Voice AI API
[voiceai-api]
enabled = true
filter = voiceai-api
logpath = /app/logs/api.log
maxretry = 5
bantime = 1800
EOF

# Create custom Fail2Ban filter for Voice AI API
cat > /etc/fail2ban/filter.d/voiceai-api.conf << EOF
# Fail2Ban filter for Voice AI Agent API

[Definition]
failregex = ^.*\[ERROR\].*Failed login attempt from <HOST>.*$
            ^.*\[ERROR\].*Invalid API key from <HOST>.*$
            ^.*\[ERROR\].*Rate limit exceeded from <HOST>.*$
            ^.*\[ERROR\].*Suspicious activity from <HOST>.*$

ignoreregex =
EOF

# Restart Fail2Ban
systemctl enable fail2ban
systemctl restart fail2ban

log "Fail2Ban configured and started"

# 5. SSH Hardening
log "5. Hardening SSH configuration..."

backup_config "/etc/ssh/sshd_config"

# SSH hardening configuration
cat >> /etc/ssh/sshd_config << EOF

# Voice AI Agent SSH Hardening
Port $SSH_PORT
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
MaxAuthTries 3
MaxSessions 2
MaxStartups 2
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
X11Forwarding no
AllowTcpForwarding no
GatewayPorts no
PermitTunnel no
Banner /etc/ssh/banner

# Allowed users (configure as needed)
# AllowUsers yourusername

# Ciphers and algorithms
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha2-256,hmac-sha2-512
KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group16-sha512,diffie-hellman-group18-sha512,diffie-hellman-group14-sha256
EOF

# Create SSH banner
cat > /etc/ssh/banner << EOF
**************************************************************************
                        AUTHORIZED ACCESS ONLY
**************************************************************************
This system is for authorized users only. All activities are monitored
and logged. Unauthorized access is strictly prohibited and will be
prosecuted to the full extent of the law.
**************************************************************************
EOF

# Restart SSH service
systemctl restart sshd

log "SSH hardening completed"

# 6. Secure Kernel Parameters
log "6. Configuring secure kernel parameters..."

backup_config "/etc/sysctl.conf"

cat >> /etc/sysctl.conf << EOF

# Voice AI Agent Security Hardening

# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1

# IPv6 security
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Memory protection
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
kernel.yama.ptrace_scope = 1

# File system hardening
fs.suid_dumpable = 0
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
EOF

# Apply sysctl settings
sysctl -p

log "Kernel parameters hardened"

# 7. Configure Automatic Updates
log "7. Configuring automatic security updates..."

backup_config "/etc/apt/apt.conf.d/50unattended-upgrades"

cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id} ESMApps:\${distro_codename}-apps-security";
    "\${distro_id} ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::Package-Blacklist {
    // Add packages to blacklist if needed
};

Unattended-Upgrade::DevRelease "false";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Remove-New-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";

Unattended-Upgrade::Mail "admin@yourdomain.com";
Unattended-Upgrade::MailOnlyOnError "true";

Unattended-Upgrade::SyslogEnable "true";
Unattended-Upgrade::SyslogFacility "daemon";
EOF

cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF

systemctl enable unattended-upgrades

log "Automatic security updates configured"

# 8. File Integrity Monitoring with AIDE
log "8. Setting up file integrity monitoring..."

# Initialize AIDE database
aide --init
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create AIDE cron job
cat > /etc/cron.d/aide << EOF
# AIDE integrity check - daily at 3:30 AM
30 3 * * * root /usr/bin/aide --check --config=/etc/aide/aide.conf | /bin/mail -s "AIDE Report" admin@yourdomain.com
EOF

log "AIDE file integrity monitoring configured"

# 9. Docker Security Hardening
log "9. Hardening Docker configuration..."

# Create Docker daemon configuration
mkdir -p /etc/docker
backup_config "/etc/docker/daemon.json"

cat > /etc/docker/daemon.json << EOF
{
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp.json",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": ["overlay2.override_kernel_check=true"],
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  }
}
EOF

# Create Docker seccomp profile
cat > /etc/docker/seccomp.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "archMap": [
    {
      "architecture": "SCMP_ARCH_X86_64",
      "subArchitectures": [
        "SCMP_ARCH_X86",
        "SCMP_ARCH_X32"
      ]
    }
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "adjtimex",
        "alarm",
        "bind",
        "brk",
        "capget",
        "capset",
        "chdir",
        "chmod",
        "chown",
        "chown32",
        "clock_getres",
        "clock_gettime",
        "clock_nanosleep",
        "close",
        "connect",
        "copy_file_range",
        "creat",
        "dup",
        "dup2",
        "dup3",
        "epoll_create",
        "epoll_create1",
        "epoll_ctl",
        "epoll_ctl_old",
        "epoll_pwait",
        "epoll_wait",
        "epoll_wait_old",
        "eventfd",
        "eventfd2",
        "execve",
        "execveat",
        "exit",
        "exit_group",
        "faccessat",
        "fadvise64",
        "fadvise64_64",
        "fallocate",
        "fanotify_mark",
        "fchdir",
        "fchmod",
        "fchmodat",
        "fchown",
        "fchown32",
        "fchownat",
        "fcntl",
        "fcntl64",
        "fdatasync",
        "fgetxattr",
        "flistxattr",
        "flock",
        "fork",
        "fremovexattr",
        "fsetxattr",
        "fstat",
        "fstat64",
        "fstatat64",
        "fstatfs",
        "fstatfs64",
        "fsync",
        "ftruncate",
        "ftruncate64",
        "futex",
        "getcpu",
        "getcwd",
        "getdents",
        "getdents64",
        "getegid",
        "getegid32",
        "geteuid",
        "geteuid32",
        "getgid",
        "getgid32",
        "getgroups",
        "getgroups32",
        "getitimer",
        "getpgid",
        "getpgrp",
        "getpid",
        "getppid",
        "getpriority",
        "getrandom",
        "getresgid",
        "getresgid32",
        "getresuid",
        "getresuid32",
        "getrlimit",
        "get_robust_list",
        "getrusage",
        "getsid",
        "getsockname",
        "getsockopt",
        "get_thread_area",
        "gettid",
        "gettimeofday",
        "getuid",
        "getuid32",
        "getxattr",
        "inotify_add_watch",
        "inotify_init",
        "inotify_init1",
        "inotify_rm_watch",
        "io_cancel",
        "ioctl",
        "io_destroy",
        "io_getevents",
        "io_setup",
        "io_submit",
        "ipc",
        "kill",
        "lchown",
        "lchown32",
        "lgetxattr",
        "link",
        "linkat",
        "listen",
        "listxattr",
        "llistxattr",
        "lremovexattr",
        "lseek",
        "lsetxattr",
        "lstat",
        "lstat64",
        "madvise",
        "memfd_create",
        "mincore",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "mlock",
        "mlock2",
        "mlockall",
        "mmap",
        "mmap2",
        "mprotect",
        "mq_getsetattr",
        "mq_notify",
        "mq_open",
        "mq_timedreceive",
        "mq_timedsend",
        "mq_unlink",
        "mremap",
        "msgctl",
        "msgget",
        "msgrcv",
        "msgsnd",
        "msync",
        "munlock",
        "munlockall",
        "munmap",
        "nanosleep",
        "newfstatat",
        "_newselect",
        "open",
        "openat",
        "pause",
        "pipe",
        "pipe2",
        "poll",
        "ppoll",
        "prctl",
        "pread64",
        "preadv",
        "prlimit64",
        "pselect6",
        "ptrace",
        "pwrite64",
        "pwritev",
        "read",
        "readahead",
        "readlink",
        "readlinkat",
        "readv",
        "recv",
        "recvfrom",
        "recvmmsg",
        "recvmsg",
        "remap_file_pages",
        "removexattr",
        "rename",
        "renameat",
        "renameat2",
        "restart_syscall",
        "rmdir",
        "rt_sigaction",
        "rt_sigpending",
        "rt_sigprocmask",
        "rt_sigqueueinfo",
        "rt_sigreturn",
        "rt_sigsuspend",
        "rt_sigtimedwait",
        "rt_tgsigqueueinfo",
        "sched_getaffinity",
        "sched_getattr",
        "sched_getparam",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getscheduler",
        "sched_rr_get_interval",
        "sched_setaffinity",
        "sched_setattr",
        "sched_setparam",
        "sched_setscheduler",
        "sched_yield",
        "seccomp",
        "select",
        "semctl",
        "semget",
        "semop",
        "semtimedop",
        "send",
        "sendfile",
        "sendfile64",
        "sendmmsg",
        "sendmsg",
        "sendto",
        "setfsgid",
        "setfsgid32",
        "setfsuid",
        "setfsuid32",
        "setgid",
        "setgid32",
        "setgroups",
        "setgroups32",
        "setitimer",
        "setpgid",
        "setpriority",
        "setregid",
        "setregid32",
        "setresgid",
        "setresgid32",
        "setresuid",
        "setresuid32",
        "setreuid",
        "setreuid32",
        "setrlimit",
        "set_robust_list",
        "setsid",
        "setsockopt",
        "set_thread_area",
        "set_tid_address",
        "setuid",
        "setuid32",
        "setxattr",
        "shmat",
        "shmctl",
        "shmdt",
        "shmget",
        "shutdown",
        "sigaltstack",
        "signalfd",
        "signalfd4",
        "sigreturn",
        "socket",
        "socketcall",
        "socketpair",
        "splice",
        "stat",
        "stat64",
        "statfs",
        "statfs64",
        "statx",
        "symlink",
        "symlinkat",
        "sync",
        "sync_file_range",
        "syncfs",
        "sysinfo",
        "tee",
        "tgkill",
        "time",
        "timer_create",
        "timer_delete",
        "timerfd_create",
        "timerfd_gettime",
        "timerfd_settime",
        "timer_getoverrun",
        "timer_gettime",
        "timer_settime",
        "times",
        "tkill",
        "truncate",
        "truncate64",
        "ugetrlimit",
        "umask",
        "uname",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
        "vfork",
        "vmsplice",
        "wait4",
        "waitid",
        "waitpid",
        "write",
        "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

# Restart Docker service
systemctl restart docker

log "Docker security hardening completed"

# 10. Log Monitoring and Analysis
log "10. Setting up log monitoring..."

# Configure logwatch
backup_config "/etc/logwatch/conf/logwatch.conf"

cat > /etc/logwatch/conf/logwatch.conf << EOF
MailTo = admin@yourdomain.com
MailFrom = logwatch@yourdomain.com
Print = No
Save = /var/cache/logwatch
Range = yesterday
Detail = Med
Service = All
mailer = "/usr/sbin/sendmail -t"
EOF

# Create custom logwatch configuration for Voice AI
mkdir -p /etc/logwatch/conf/services
cat > /etc/logwatch/conf/services/voiceai.conf << EOF
Title = "Voice AI Agent"
LogFile = /app/logs/*.log
*OnlyService = voiceai
*RemoveHeaders = 
EOF

log "Log monitoring configured"

# 11. Security Monitoring Script
log "11. Creating security monitoring script..."

cat > /usr/local/bin/security-check.sh << 'EOF'
#!/bin/bash
# Security Check Script for Voice AI Agent

LOG_FILE="/var/log/security-check.log"

echo "$(date): Starting security check" >> "$LOG_FILE"

# Check for failed login attempts
FAILED_LOGINS=$(grep "Failed password" /var/log/auth.log | wc -l)
if [ "$FAILED_LOGINS" -gt 10 ]; then
    echo "$(date): WARNING: $FAILED_LOGINS failed login attempts detected" >> "$LOG_FILE"
fi

# Check for rootkit
if command -v rkhunter &> /dev/null; then
    rkhunter --check --sk --nocolors >> "$LOG_FILE" 2>&1
fi

# Check for malware
if command -v chkrootkit &> /dev/null; then
    chkrootkit >> "$LOG_FILE" 2>&1
fi

# Check open ports
OPEN_PORTS=$(netstat -tuln | grep LISTEN | wc -l)
echo "$(date): Open ports: $OPEN_PORTS" >> "$LOG_FILE"

# Check disk usage
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "$(date): WARNING: Disk usage is high: ${DISK_USAGE}%" >> "$LOG_FILE"
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ "$MEM_USAGE" -gt 90 ]; then
    echo "$(date): WARNING: Memory usage is high: ${MEM_USAGE}%" >> "$LOG_FILE"
fi

# Check Docker container status
RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}" | grep -c voiceai || echo "0")
echo "$(date): Running Voice AI containers: $RUNNING_CONTAINERS" >> "$LOG_FILE"

# Check fail2ban status
BANNED_IPS=$(fail2ban-client status | grep "Jail list:" | sed 's/.*Jail list://' | wc -w)
echo "$(date): Fail2ban banned IPs: $BANNED_IPS" >> "$LOG_FILE"

echo "$(date): Security check completed" >> "$LOG_FILE"
EOF

chmod +x /usr/local/bin/security-check.sh

# Add security check to cron
cat > /etc/cron.d/security-check << EOF
# Security check - every 6 hours
0 */6 * * * root /usr/local/bin/security-check.sh
EOF

log "Security monitoring script created"

# 12. SSL/TLS Configuration
log "12. Preparing SSL/TLS configuration..."

# Create SSL directory
mkdir -p /etc/nginx/ssl

# Create self-signed certificate (replace with Let's Encrypt in production)
if [ ! -f /etc/nginx/ssl/selfsigned.crt ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/selfsigned.key \
        -out /etc/nginx/ssl/selfsigned.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"
    
    log "Self-signed SSL certificate created (replace with Let's Encrypt)"
fi

# Create SSL configuration snippet
cat > /etc/nginx/ssl/ssl-params.conf << EOF
# SSL Configuration for Voice AI Agent
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers on;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_ecdh_curve secp384r1;
ssl_session_timeout 10m;
ssl_session_cache shared:SSL:10m;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# Security headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' wss: ws:; frame-ancestors 'none';";
EOF

log "SSL/TLS configuration prepared"

# 13. Final Security Report
log "13. Generating security report..."

cat > /var/log/security-hardening-report.txt << EOF
Voice AI Agent Security Hardening Report
========================================
Date: $(date)
Hostname: $(hostname)
OS: $(lsb_release -d | cut -f2)

Security Measures Implemented:
==============================

1. System Updates
   - All packages updated to latest versions
   - Automatic security updates enabled

2. Firewall (UFW)
   - Default deny incoming policy
   - SSH port $SSH_PORT allowed
   - HTTP/HTTPS ports allowed
   - Monitoring ports allowed
   - Docker networks allowed

3. Intrusion Detection (Fail2Ban)
   - SSH protection enabled
   - Nginx protection enabled
   - Custom Voice AI API protection
   - Ban time: 1 hour for most services

4. SSH Hardening
   - Root login disabled
   - Password authentication disabled
   - Key-based authentication only
   - Connection limits implemented
   - Strong ciphers enforced

5. Kernel Security
   - Network security parameters tuned
   - Memory protection enabled
   - File system hardening applied

6. File Integrity Monitoring (AIDE)
   - Database initialized
   - Daily integrity checks scheduled

7. Docker Security
   - Security profiles applied
   - Resource limits enforced
   - Logging configured

8. Log Monitoring
   - Logwatch configured
   - Security monitoring script created
   - Regular security checks scheduled

9. SSL/TLS
   - Strong cipher suites configured
   - Security headers implemented
   - Certificate management prepared

Current Status:
==============
- UFW Status: $(ufw status | head -n 1)
- Fail2Ban Status: $(systemctl is-active fail2ban)
- SSH Status: $(systemctl is-active sshd)
- Docker Status: $(systemctl is-active docker)
- Unattended Upgrades: $(systemctl is-active unattended-upgrades)

Next Steps:
===========
1. Configure Let's Encrypt SSL certificates
2. Set up log aggregation and SIEM
3. Implement network monitoring
4. Configure backup encryption
5. Set up incident response procedures
6. Regular security audits and penetration testing

Security Contacts:
==================
System Administrator: admin@yourdomain.com
Security Team: security@yourdomain.com
EOF

log "Security hardening report generated: /var/log/security-hardening-report.txt"

# Final summary
echo ""
echo -e "${GREEN}Security hardening completed successfully!${NC}"
echo "============================================="
echo "Key security measures implemented:"
echo "  ✓ System updates and automatic security updates"
echo "  ✓ UFW firewall with restrictive rules"
echo "  ✓ Fail2Ban intrusion detection"
echo "  ✓ SSH hardening and key-based authentication"
echo "  ✓ Kernel security parameters optimized"
echo "  ✓ File integrity monitoring with AIDE"
echo "  ✓ Docker security hardening"
echo "  ✓ Log monitoring and analysis"
echo "  ✓ SSL/TLS configuration prepared"
echo "  ✓ Security monitoring scripts deployed"
echo ""
echo -e "${YELLOW}Important next steps:${NC}"
echo "1. Replace self-signed SSL with Let's Encrypt:"
echo "   certbot --nginx -d your-domain.com"
echo ""
echo "2. Configure email notifications for alerts"
echo ""
echo "3. Review and test all security measures"
echo ""
echo "4. Set up regular security audits"
echo ""
echo -e "${BLUE}Security reports and logs:${NC}"
echo "  - Hardening report: /var/log/security-hardening-report.txt"
echo "  - Security check log: /var/log/security-check.log"
echo "  - Fail2Ban log: /var/log/fail2ban.log"
echo "  - UFW log: /var/log/ufw.log"
echo ""
echo -e "${GREEN}Security hardening completed at: $(date)${NC}"

exit 0