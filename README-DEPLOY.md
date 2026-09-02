# Vercel 部署说明

将本目录中的 `api/`、`knowledge/`、`package.json` 和 `vercel.json` 放到 GitHub 仓库根目录；保留现有的 `tools/` 静态工具目录。然后在 Vercel 导入该 GitHub 仓库。

在 Vercel 项目 Settings → Environment Variables 配置：

- `DEEPSEEK_API_KEY`：仅服务端保存，不写入网页或 GitHub。
- `APP_ACCESS_PASSWORD`：访问问答模块的长随机密码；不要使用 GitHub、邮箱或 DeepSeek 的登录密码。
- `DEEPSEEK_MODEL`：默认 `deepseek-v4-flash`。
- `MAX_REQUESTS_PER_HOUR`：单个 IP 的每小时上限，默认 20。

GitHub Pages 可继续用于纯决策工具；带问答模块的正式入口应分享 Vercel 域名，例如：
`https://你的项目.vercel.app/tools/scaphoid-decision-tool/scaphoid-decision-tool.html`。

当前限流是无状态的基础保护，适合低量私密使用。若公开分享给多人，应在 Vercel Firewall 中增加限流规则，或接入 Upstash Redis 实现跨实例的持久限流。访问密码只能减少随意访问，不能替代密钥保密和限流。
