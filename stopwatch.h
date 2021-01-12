#include <chrono>

class Stopwatch
{
public:
	void Start();
	void Stop();
	double Time() const;
private:
	std::chrono::steady_clock::time_point startPoint;
	std::chrono::steady_clock::time_point endPoint;
};

inline void Stopwatch::Start()
{
	// Record start time
	startPoint = std::chrono::high_resolution_clock::now();
}

inline void Stopwatch::Stop()
{
	// Record start time
	endPoint = std::chrono::high_resolution_clock::now();
}

inline double Stopwatch::Time() const
{
	const auto time = std::chrono::duration<double, std::milli>(endPoint-startPoint);
	return time.count();
}
