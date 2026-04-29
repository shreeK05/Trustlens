FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
COPY frontend/ .
RUN npm ci && npm run build

FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
