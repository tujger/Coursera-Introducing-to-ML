package com.pantheoninc.sparktest;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class Timer {
    long start;
    List<Long> ticks;



    Timer() {
        ticks = new ArrayList<>();
    }

    Timer start() {
        start = Calendar.getInstance().getTimeInMillis();
        ticks.add(start);
        return this;
    }

    Long tick() {
        long now = Calendar.getInstance().getTimeInMillis();
        ticks.add(now);
        long delta = now - start;
        start = now;
        return delta;
    }

    Long stop() {
        long now = Calendar.getInstance().getTimeInMillis();
        ticks.add(now);
        long delta = now - ticks.get(0);
        return delta;
    }

    Timer startLog() {
        start();
        System.out.println("Timer started at: " + new Date(start).toString());
        return this;
    }

    void tickLog() {
        System.out.println("Timer [" + ticks.size() + "]: " + tick() + " ms");
    }

    void tickLog(String mark) {
        System.out.println("Timer [" + ticks.size() + ":" + mark + "]: " + tick() + " ms");
    }

    void stopLog() {
        stop();
        System.out.println("Timer stopped at: " + new Date(start).toString() + " with " + (ticks.size() - 2) + " tick(s) and total time: " + (ticks.get(ticks.size() - 1) - ticks.get(0)) + " ms");
    }
}
