#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>

const double G = 6.67430e-11;

struct Body {
    double mass;
    double x, y;
    double vx, vy;
    double ax, ay;
    sf::CircleShape shape;
    sf::Color trail_color;
};

int main() {
    const int N = 50;            
    const int window_size = 8000;
    const double scale = 1e2;
    const double dt = 60*60;     

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(window_size, window_size)), "Sistema solar simplificado");

    std::vector<Body> bodies(N);

    bodies[0].mass = 2e30;       
    bodies[0].x = 0;
    bodies[0].y = 0;
    bodies[0].vx = 0;
    bodies[0].vy = 0;
    bodies[0].shape = sf::CircleShape(5.f);
    bodies[0].shape.setFillColor(sf::Color::Yellow);
    bodies[0].trail_color = sf::Color(255, 255, 0, 150); 

    double base_radius = 5e12;
    double radius_step = 3e14;

    sf::Color trail_colors[5] = {
        sf::Color(0, 0, 255, 150),      
        sf::Color(255, 0, 0, 150),      
        sf::Color(0, 255, 0, 150),      
        sf::Color(255, 0, 255, 150),    
        sf::Color(0, 255, 255, 150)     
    };

    for (int i = 1; i < N; ++i) {
        bodies[i].mass = 1e24 + i * 1e23;  
        bodies[i].x = base_radius + radius_step * (i - 1);
        bodies[i].y = 0;
        double r = bodies[i].x;
        double v = std::sqrt(G * bodies[0].mass / r);
        bodies[i].vx = 0;
        bodies[i].vy = v;
        bodies[i].shape = sf::CircleShape(5.f);

        sf::Color color;
        switch (i % 5) {
            case 0: color = sf::Color::Blue; break;
            case 1: color = sf::Color::Red; break;
            case 2: color = sf::Color::Green; break;
            case 3: color = sf::Color::Magenta; break;
            case 4: color = sf::Color::Cyan; break;
        }
        bodies[i].shape.setFillColor(color);
        bodies[i].trail_color = trail_colors[i % 5];
    }
    const int max_trail_points = 500;
    std::vector<std::vector<sf::Vector2f>> trails(N);

    auto compute_accelerations = [&](std::vector<Body>& b) {
        for (int i = 0; i < N; ++i) {
            b[i].ax = 0;
            b[i].ay = 0;
        }
        for (int i = 1; i < N; ++i) {
            double dx = b[0].x - b[i].x;
            double dy = b[0].y - b[i].y;
            double distSq = dx*dx + dy*dy + 1e10; 
            double dist = std::sqrt(distSq);
            double force = G * b[0].mass / distSq;

            b[i].ax = force * dx / dist;
            b[i].ay = force * dy / dist;

            b[0].ax -= force * dx / dist * (b[i].mass / b[0].mass);
            b[0].ay -= force * dy / dist * (b[i].mass / b[0].mass);
        }
    };
    compute_accelerations(bodies);
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        for (int i = 0; i < N; ++i) {
            bodies[i].x += bodies[i].vx * dt + 0.5 * bodies[i].ax * dt * dt;
            bodies[i].y += bodies[i].vy * dt + 0.5 * bodies[i].ay * dt * dt;
        }

        std::vector<std::pair<double,double>> old_acc(N);
        for (int i = 0; i < N; ++i) {
            old_acc[i].first = bodies[i].ax;
            old_acc[i].second = bodies[i].ay;
        }
        compute_accelerations(bodies);
        
        for (int i = 0; i < N; ++i) {
            bodies[i].vx += 0.5 * (old_acc[i].first + bodies[i].ax) * dt;
            bodies[i].vy += 0.5 * (old_acc[i].second + bodies[i].ay) * dt;
        }

        for (int i = 0; i < N; ++i) {
            float x_pix = window_size / 2 + bodies[i].x / scale;
            float y_pix = window_size / 2 + bodies[i].y / scale;
            sf::Vector2f current_pos(x_pix, y_pix);

            trails[i].push_back(current_pos);
            if (trails[i].size() > max_trail_points) {
                trails[i].erase(trails[i].begin());
            }
        }

        window.clear(sf::Color::Black);
        
        for (int i = 0; i < N; ++i) {
            if (trails[i].size() < 2) continue;

            std::vector<sf::Vertex> line_points;
            line_points.reserve(trails[i].size());

            for (const auto& pos : trails[i]) {
                line_points.emplace_back(pos, bodies[i].trail_color);
            }
            window.draw(&line_points[0], line_points.size(), sf::LineStrip);
        }
        
        for (auto& body : bodies) {
            float x_pix = window_size / 2 + body.x / scale;
            float y_pix = window_size / 2 + body.y / scale;
            body.shape.setPosition(sf::Vector2f(x_pix - body.shape.getRadius(), y_pix - body.shape.getRadius()));
            window.draw(body.shape);
        }
        window.display();
    }
    return 0;
}

// This code simulates a simplified solar system with a central body and several orbiting bodies.
// The bodies are represented as circles, and their positions and velocities are updated based on gravitational interactions.
// Trails are drawn to visualize the paths of the orbiting bodies. The simulation runs in a window using SFML for graphics.
// The code includes basic physics calculations for gravitational forces and motion updates, and it uses a vector to store the positions of the trails for each body.
// The simulation continues until the window is closed, allowing for real-time visualization of the system's dynamics.
// To execute the code, use in vsc terminal: 
// g++ -std=c++17 simulation_hmxb.cpp \
  -I/usr/local/homebrew/opt/sfml/include \
  -L/usr/local/homebrew/opt/sfml/lib \
  -lsfml-graphics -lsfml-window -lsfml-system \
  -o simulation_hmxb
// ./simulation_hmxb